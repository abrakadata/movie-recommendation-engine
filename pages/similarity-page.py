import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from utils import apply_filters, display_results, load_data, load_model, render_sidebar

model = load_model()
df, embeddings = load_data()
controls = render_sidebar(df, "Movie similarity")

st.header("Movie similarity")

selected_title = st.selectbox(
    "Pick a movie you like",
    options=sorted(df["title"].tolist()),
    index=None,
    placeholder="Search for a title…",
)

if selected_title is None:
    st.stop()

idx = df[df["title"] == selected_title].index[0]

row = df.iloc[idx]
_box = (
    "background:#1a3a52;border-left:4px solid #4a9eda;"
    "padding:0.75rem 1rem;border-radius:0.375rem;margin-bottom:0.5rem;color:#e0e0e0"
)
st.markdown(
    f'<div style="{_box}">🎬 &nbsp;<strong>{row["title"]}</strong><br>'
    f'<strong>Tags:</strong> {row["tags"]}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="{_box}"><strong>Synopsis:</strong> {row["plot_synopsis"][:200]}…</div>',
    unsafe_allow_html=True,
)

seed_vec = embeddings[idx].reshape(1, -1)
content_scores = cosine_similarity(seed_vec, embeddings)[0]
hybrid = (
    controls["similarity_weight"] * content_scores
    + (1 - controls["similarity_weight"]) * df["pop_score"].values
)
hybrid[idx] = 0.0
hybrid = apply_filters(df, hybrid, controls["tag_filter"])

top_idx = hybrid.argsort()[::-1][: controls["n_results"]]

if hybrid[top_idx].max() == 0.0:
    st.info("No movies matched your filters. Try relaxing the tag filter.")
    st.stop()

top_idx = top_idx[df["pop_score"].values[top_idx].argsort()[::-1]]

display_results(df, top_idx, content_scores[top_idx])
