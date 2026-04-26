import ast
import base64
import os

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "movies_metadata.csv"
EMB_CACHE_PATH = "embeddings_cache.npy"

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[["title", "genres", "overview", "popularity", "vote_average", "vote_count"]]
    df = df.dropna(subset=["overview", "genres"])
    df = df[df["overview"].str.strip() != ""]
    df = df[df["genres"].str.strip() != ""]
    df["genres"] = df["genres"].apply(_parse_genres)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    df["vote_average"] = df["vote_average"].fillna(df["vote_average"].median())
    df["vote_count"] = df["vote_count"].fillna(df["vote_count"].median())
    df = df.reset_index(drop=True)
    build_popularity_scores(df)
    return df


def _parse_genres(raw):
    try:
        items = ast.literal_eval(raw)
        return [g["name"] for g in items if "name" in g]
    except (ValueError, SyntaxError):
        return []


def build_popularity_scores(df):
    log_counts = np.log1p(df["vote_count"])
    df["pop_score"] = (df["vote_average"] / 10) * (log_counts / log_counts.max())


def build_feature_text(df):
    def row_text(row):
        genres = " ".join(row["genres"])
        return f"{genres} {row['overview']}"
    return df.apply(row_text, axis=1)


def build_embedding_matrix(texts):
    if os.path.exists(EMB_CACHE_PATH):
        cached = np.load(EMB_CACHE_PATH)
        if len(cached) == len(texts):
            return cached
        os.remove(EMB_CACHE_PATH)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts_list = texts.tolist()
    batch_size = 64
    n = len(texts_list)
    batches = range(0, n, batch_size)

    bar = st.progress(0, text="Building embeddings — first run only…")
    chunks = []
    for i in batches:
        chunks.append(model.encode(texts_list[i:i + batch_size], show_progress_bar=False))
        bar.progress(min((i + batch_size) / n, 1.0))
    bar.empty()

    matrix = np.vstack(chunks)
    np.save(EMB_CACHE_PATH, matrix)
    return matrix


def get_content_scores(seed_indices, matrix):
    seed_vectors = matrix[seed_indices]
    scores = cosine_similarity(seed_vectors, matrix)
    return scores.mean(axis=0)


def recommend(seed_titles, df, matrix, content_weight=0.75, pop_weight=0.25, n=5):
    seed_titles = list(dict.fromkeys(seed_titles))  # deduplicate, preserve order
    seed_indices = []
    exclude_indices = []
    not_found = []
    for title in seed_titles:
        matches = df[df["title"].str.contains(title, case=False, na=False, regex=False)]
        if matches.empty:
            not_found.append(title)
        else:
            seed_indices.append(matches.index[0])
            exclude_indices.extend(matches.index.tolist())

    if not seed_indices:
        return pd.DataFrame(), not_found

    content_scores = get_content_scores(seed_indices, matrix)
    hybrid = content_weight * content_scores + pop_weight * df["pop_score"].values

    for idx in exclude_indices:
        hybrid[idx] = 0.0

    seed_genres = set(g for idx in seed_indices for g in df.iloc[idx]["genres"])

    top_idx = hybrid.argsort()[::-1][:n]
    result = df.iloc[top_idx][["title", "genres", "vote_average"]].copy()
    result["score"] = hybrid[top_idx]
    result = result.reset_index(drop=True)
    result["explanation"] = result.apply(lambda row: explain(row, seed_genres), axis=1)
    return result, not_found


def explain(row, seed_genres):
    shared = [g for g in row["genres"] if g in seed_genres]
    if shared:
        bullet1 = ", ".join(shared)
    else:
        bullet1 = "Similar themes and storyline"

    try:
        rating = float(row["vote_average"])
        bullet2 = f"Rated {rating:.1f}/10 by audiences"
    except (TypeError, ValueError):
        bullet2 = "Popular with audiences"

    return [bullet1, bullet2]


def _display_table(results):
    is_fallback = "explanation" not in results.columns

    table = pd.DataFrame({
        "Title": results["title"],
        "Genre": (
            results["explanation"].apply(lambda e: e[0])
            if not is_fallback
            else results["genres"].apply(lambda g: ", ".join(g))
        ),
        "Rating": results["vote_average"].apply(lambda v: round(float(v), 1)),
    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Genre": st.column_config.TextColumn("Genre", width="medium"),
            "Rating": st.column_config.ProgressColumn(
                "Rating",
                min_value=0,
                max_value=10,
                format="%.1f",
                width="small",
            ),
        },
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
div.stButton > button {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.4rem 1.2rem;
}
div.stButton > button:hover {
    background-color: #2563eb;
    color: white;
}
[data-testid="collapsedControl"] {
    display: none;
}
div[role="radiogroup"] label {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

title_slot = st.empty()
title_slot.title("Movie Recommendations" if "results" in st.session_state else "Enter some movies you like")

df = load_data(DATA_PATH)
if "emb" not in st.session_state:
    title_slot.title("Updating Embeddings…")
    st.session_state.emb = build_embedding_matrix(build_feature_text(df))
    title_slot.title("Movie Recommendations" if "results" in st.session_state else "Enter some movies you like")
emb = st.session_state.emb

_PRIORITY_OPTIONS = {
    "Similarity 75% — Popularity 25%": (0.75, 0.25),
    "Popularity 75% — Similarity 25%": (0.25, 0.75),
    "50% Similarity — 50% Popularity": (0.50, 0.50),
}

with st.sidebar:
    st.markdown(
        '<p style="font-size:2.6rem; font-weight:800; color:#198754; line-height:1.2; margin-bottom:1.5rem; text-align:center;">Priority</p>',
        unsafe_allow_html=True,
    )
    priority = st.radio(
        label="Weighting",
        options=list(_PRIORITY_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown(
        '<p style="color:#fafafa; font-size:0.85rem; margin-top:3rem;"><strong>Note</strong> — Choose how much weight to give genre/theme similarity vs. overall popularity when ranking results.</p>',
        unsafe_allow_html=True,
    )

content_w, pop_w = _PRIORITY_OPTIONS[priority]

if "results" not in st.session_state:
    user_input = st.text_input("Enter 3–5 movies (comma-separated)")

    if st.button("Find Movies"):
        if not user_input.strip():
            st.warning("Please enter at least one movie title.")
        else:
            seed_titles = [t.strip() for t in user_input.split(",") if t.strip()]
            results, not_found = recommend(seed_titles, df, emb, content_weight=content_w, pop_weight=pop_w)
            st.session_state.results = results
            st.session_state.not_found = not_found
            st.rerun()
else:
    results = st.session_state.results
    not_found = st.session_state.not_found

    if not_found:
        st.warning(f"Couldn't find: {', '.join(not_found)}. Skipped.")

    if results.empty:
        st.warning("We couldn't match your titles. Here are some well-rated films.")
        _display_table(df.nlargest(5, "pop_score"))
    else:
        if len(results) < 5:
            st.info(f"Only {len(results)} recommendation(s) found.")
        _display_table(results)

    if st.button("Restart"):
        del st.session_state.results
        del st.session_state.not_found
        st.rerun()
