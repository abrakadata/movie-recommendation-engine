import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from utils import apply_filters, display_results, load_data, load_model, render_sidebar

model = load_model()
df, embeddings = load_data()
controls = render_sidebar(df)

st.header("Free-text search")

query_text = st.text_area(
    "Describe the kind of movie you want to watch",
    placeholder="A psychological thriller where nothing is what it seems",
    height=100,
)

if st.button("Find Movies"):
    if not query_text.strip():
        st.warning("Please enter a description.")
        st.stop()

    query_vec = model.encode([query_text])
    content_scores = cosine_similarity(query_vec, embeddings)[0]
    hybrid = (
        controls["similarity_weight"] * content_scores
        + (1 - controls["similarity_weight"]) * df["pop_score"].values
    )
    hybrid = apply_filters(df, hybrid, controls["tag_filter"], controls["min_similarity"])

    top_idx = hybrid.argsort()[::-1][: controls["n_results"]]

    if hybrid[top_idx].max() == 0.0:
        st.info("No movies matched your filters. Try relaxing the tag filter or lowering the similarity threshold.")
        st.stop()

    display_results(df, top_idx, content_scores[top_idx])
