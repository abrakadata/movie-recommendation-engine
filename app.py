import ast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "movies_metadata.csv"


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


@st.cache_data
def build_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    return vectorizer.fit_transform(texts)


def get_content_scores(seed_indices, tfidf_matrix):
    seed_vectors = tfidf_matrix[seed_indices]
    scores = cosine_similarity(seed_vectors, tfidf_matrix)
    return scores.mean(axis=0)


def recommend(seed_titles, df, tfidf_matrix, n=5):
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

    content_scores = get_content_scores(seed_indices, tfidf_matrix)
    hybrid = 0.75 * content_scores + 0.25 * df["pop_score"].values

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
        bullet1 = "Shares genres: " + ", ".join(shared)
    else:
        bullet1 = "Similar themes and storyline"

    try:
        rating = float(row["vote_average"])
        bullet2 = f"Rated {rating:.1f}/10 by audiences"
    except (TypeError, ValueError):
        bullet2 = "Popular with audiences"

    return [bullet1, bullet2]


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("Movie Recommender")

df = load_data(DATA_PATH)
tfidf = build_tfidf_matrix(build_feature_text(df))

user_input = st.text_input("Enter 3–5 movies you like (comma-separated)")

if st.button("Find Movies"):
    if not user_input.strip():
        st.warning("Please enter at least one movie title.")
    else:
        seed_titles = [t.strip() for t in user_input.split(",") if t.strip()]
        results, not_found = recommend(seed_titles, df, tfidf)

        if not_found:
            st.warning(f"Couldn't find: {', '.join(not_found)}. Skipped.")

        if results.empty:
            st.warning("We couldn't match your titles. Here are some well-rated films.")
            fallback = df.nlargest(5, "pop_score")
            for _, row in fallback.iterrows():
                st.markdown(f"**{row['title']}**")
                st.write(f"Genres: {', '.join(row['genres'])}")
                st.write(f"Rating: {row['vote_average']:.1f}/10")
                st.divider()
        else:
            if len(results) < 5:
                st.info(f"Only {len(results)} recommendation(s) found.")
            for _, row in results.iterrows():
                st.markdown(f"**{row['title']}**")
                st.write(f"Genres: {', '.join(row['genres'])}")
                st.write(f"Rating: {row['vote_average']:.1f}/10")
                for bullet in row["explanation"]:
                    st.write(f"• {bullet}")
                st.divider()
