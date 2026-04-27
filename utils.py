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


def render_sidebar(df):
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] {
        max-width: 95vw !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    all_tags = sorted({
        tag.strip()
        for tags in df["tags"]
        for tag in tags.split(",")
        if tag.strip()
    })
    with st.sidebar:
        st.markdown("## 🎬 Movie Recommender")
        n_results = st.slider("Number of results", min_value=3, max_value=15, value=5)
        tag_filter = st.multiselect("Tag filter", options=all_tags)
        similarity_weight = st.slider("Similarity weight (vs. popularity)",
                                      min_value=0.0, max_value=1.0,
                                      value=0.75, step=0.05)
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
        "Title":    rows["title"].values,
        "Tags":     rows["tags"].values,
        "Match":    [format_match(s) for s in scores],
        "Synopsis": [s[:200] + "…" for s in rows["plot_synopsis"].values],
    })
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title":    st.column_config.TextColumn("Title",    width="medium"),
            "Tags":     st.column_config.TextColumn("Tags",     width="medium"),
            "Match":    st.column_config.TextColumn("Match",    width="small"),
            "Synopsis": st.column_config.TextColumn("Synopsis", width="large"),
        },
    )
