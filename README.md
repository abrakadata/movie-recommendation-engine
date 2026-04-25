# Movie Recommender

A content-based movie recommender that combines semantic similarity (sentence embeddings) with a popularity prior to suggest films based on titles you already like.

## Setup

```bash
pip install streamlit pandas scikit-learn sentence-transformers
```

Place `movies_metadata.csv` (TMDB 5000 dataset) in the project root.

## Run

```bash
streamlit run app.py
```

## Dataset

Download `tmdb_5000_movies.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), rename it to `movies_metadata.csv`, and place it in the same folder as `app.py`.

## How it works

Each movie is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2` (sentence-transformers). Recommendations are scored by blending cosine similarity against your seed titles (75%) with a Bayesian popularity score derived from vote average and vote count (25%). The model is downloaded once (~90 MB) and cached.
