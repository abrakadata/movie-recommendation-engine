# Movie Recommendation Engine

A content-based movie recommender that blends semantic similarity (sentence-transformer embeddings) with a popularity prior to suggest films based on titles you already like.

## Setup

```bash
pip install streamlit pandas scikit-learn sentence-transformers
```

Place `movies_metadata.csv` (TMDB dataset, ~4,775 movies) in the project root.

## Run

```bash
streamlit run app.py
```

On first launch the app builds a `embeddings_cache.npy` file (~few seconds). Subsequent launches load it instantly.

## Dataset

Download `tmdb_5000_movies.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), rename it to `movies_metadata.csv`, and place it in the same folder as `app.py`.

## How it works

Each movie is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2` (sentence-transformers). Recommendations are scored by blending cosine similarity against your seed titles with a Bayesian popularity score derived from vote average and vote count. The model is downloaded once (~90 MB) and cached locally.

The blending ratio is user-configurable via the sidebar:

| Option | Similarity | Popularity |
|---|---|---|
| Default | 75% | 25% |
| Popularity-first | 25% | 75% |
| Balanced | 50% | 50% |

## UI Features

- Sidebar with non-collapsible Recommendation Priority selector
- Results displayed as an interactive table with Title, Genres, Rating (progress bar), and Why columns
- Embedding progress bar on first launch
- Restart button to reset without refreshing the page
