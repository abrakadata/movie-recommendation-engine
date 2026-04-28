import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

EMB_PATH = "embeddings.npy"
PKL_PATH = "movies.pkl"


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_data():
    if not (
        __import__("os").path.exists(EMB_PATH)
        and __import__("os").path.exists(PKL_PATH)
    ):
        st.error("Run python precompute.py first.")
        st.stop()
    df = pd.read_pickle(PKL_PATH)
    embeddings = np.load(EMB_PATH)
    return df, embeddings


def render_sidebar(df, current_page="Search"):
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] {
        max-width: 95vw !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] [data-id="placeholder"] {
        color: rgba(255, 255, 255, 0.25) !important;
    }
    [data-testid="stButton"] > button {
        background-color: #0d6efd !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.375rem !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease-in-out !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #0b5ed7 !important;
        color: #ffffff !important;
    }
    [data-testid="stButton"] > button:active {
        background-color: #0a58ca !important;
    }
    </style>
    """, unsafe_allow_html=True)
    all_tags = sorted({
        tag.strip()
        for tags in df["tags"]
        for tag in tags.split(",")
        if tag.strip()
    })
    _priority_options = {
        "Similarity 75% — Popularity 25%": 0.75,
        "Popularity 75% — Similarity 25%": 0.25,
        "50% Similarity — 50% Popularity": 0.50,
    }
    with st.sidebar:
        st.title("Settings")
        mode = st.radio(
            "App mode",
            options=["Search", "Movie similarity"],
            index=0 if current_page == "Search" else 1,
        )
        if mode != current_page:
            st.switch_page(
                "pages/free-search-page.py" if mode == "Search"
                else "pages/similarity-page.py"
            )
        st.divider()
        priority_label = st.radio("Priority", options=list(_priority_options.keys()))
        similarity_weight = _priority_options[priority_label]
        st.divider()
        n_results = st.slider("Number of results", min_value=3, max_value=15, value=5)
        tag_filter = st.multiselect("Tag filter", options=all_tags)
    return {
        "n_results": n_results,
        "tag_filter": tag_filter,
        "similarity_weight": similarity_weight,
    }


def format_match(score):
    n = round(score * 100)
    if score >= 0.75:
        return f"🟢 {n}%"
    if score >= 0.50:
        return f"🟡 {n}%"
    return f"🔴 {n}%"


def apply_filters(df, content_scores, tag_filter):
    scores = content_scores.copy()
    if tag_filter:
        for i, row_tags in enumerate(df["tags"]):
            if not any(t.lower() in row_tags.lower() for t in tag_filter):
                scores[i] = 0.0
    return scores


def display_results(df, indices, scores):
    rows = df.iloc[indices]
    table = pd.DataFrame({
        "Title":      rows["title"].values,
        "Tags":       rows["tags"].values,
        "Match":      [format_match(s) for s in scores],
        "Popularity": [f"{round(s * 100)}%" for s in rows["pop_score"].values],
        "Synopsis":   [s[:200] + "…" for s in rows["plot_synopsis"].values],
    })
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title":      st.column_config.TextColumn("Title",      width="medium"),
            "Tags":       st.column_config.TextColumn("Tags",       width="medium"),
            "Match":      st.column_config.TextColumn("Match",      width="small"),
            "Popularity": st.column_config.TextColumn("Popularity", width="small"),
            "Synopsis":   st.column_config.TextColumn("Synopsis",   width="large"),
        },
    )
