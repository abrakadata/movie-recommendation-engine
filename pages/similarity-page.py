import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from utils import apply_filters, display_results, load_data, load_model, render_sidebar

model = load_model()
df, embeddings = load_data()
controls = render_sidebar(df)

st.header("Movie similarity")

selected_title = st.selectbox("Pick a movie you like", options=sorted(df["title"].tolist()))

idx = df[df["title"] == selected_title].index[0]

row = df.iloc[idx]
st.info(f"🎬  {row['title']}\nTags: {row['tags']}\n{row['plot_synopsis'][:200]}…")

seed_vec = embeddings[idx].reshape(1, -1)
content_scores = cosine_similarity(seed_vec, embeddings)[0]
hybrid = (
    controls["similarity_weight"] * content_scores
    + (1 - controls["similarity_weight"]) * df["pop_score"].values
)
hybrid[idx] = 0.0
hybrid = apply_filters(df, hybrid, controls["tag_filter"], controls["min_similarity"])

top_idx = hybrid.argsort()[::-1][: controls["n_results"]]

if hybrid[top_idx].max() == 0.0:
    st.info("No movies matched your filters. Try relaxing the tag filter or lowering the similarity threshold.")
    st.stop()

display_results(df, top_idx, content_scores[top_idx])
