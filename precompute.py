import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MPST_PATH = "mpst_full_data.csv"
TMDB_PATH = "movies_metadata.csv"
EMB_OUT = "embeddings.npy"
PKL_OUT = "movies.pkl"


def main():
    t0 = time.time()

    # --- Load and clean MPST ---
    print("Loading dataset...")
    if not os.path.exists(MPST_PATH):
        print(f"Error: {MPST_PATH} not found.")
        raise SystemExit(1)

    df = pd.read_csv(MPST_PATH)

    # 1.2
    df = df.dropna(subset=["plot_synopsis", "title"])

    # 1.3
    df = df.drop_duplicates(subset=["imdb_id"], keep="first")

    # 1.4
    df["title"] = df["title"].str.strip()
    df["tags"] = df["tags"].fillna("").str.strip()

    # --- Join TMDB popularity ---
    # 1.5
    tmdb = pd.read_csv(TMDB_PATH, low_memory=False)[["title", "vote_average", "vote_count"]]
    tmdb["vote_average"] = pd.to_numeric(tmdb["vote_average"], errors="coerce")
    tmdb["vote_count"] = pd.to_numeric(tmdb["vote_count"], errors="coerce")

    # 1.6
    df["_title_key"] = df["title"].str.lower().str.strip()
    tmdb["_title_key"] = tmdb["title"].str.lower().str.strip()

    # 1.7
    tmdb_lookup = (
        tmdb.dropna(subset=["_title_key"])
        .drop_duplicates(subset=["_title_key"])[["_title_key", "vote_average", "vote_count"]]
    )
    df = df.merge(tmdb_lookup, on="_title_key", how="left")

    # 1.8
    tmdb_median = tmdb["vote_average"].median()
    df["vote_average"] = df["vote_average"].fillna(tmdb_median)
    df["vote_count"] = df["vote_count"].fillna(0)

    # 1.9
    log_counts = np.log1p(df["vote_count"])
    df["pop_score"] = (df["vote_average"] / 10) * (log_counts / log_counts.max())

    # 1.10 — drop join artifacts, keep MPST columns + pop_score
    df = df.drop(columns=["_title_key", "vote_average", "vote_count"])

    # 1.11
    df = df.reset_index(drop=True)

    # --- Encode ---
    print(f"Encoding {len(df):,} synopses with all-MiniLM-L6-v2...")

    # 1.12
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1.13
    embeddings = model.encode(
        df["plot_synopsis"].tolist(),
        batch_size=64,
        show_progress_bar=True,
    )

    # 1.14
    embeddings = embeddings.astype(np.float32)
    np.save(EMB_OUT, embeddings)

    # 1.15
    df.to_pickle(PKL_OUT)

    # 1.16
    emb_mb = os.path.getsize(EMB_OUT) / 1e6
    pkl_mb = os.path.getsize(PKL_OUT) / 1e6
    elapsed = time.time() - t0

    print(f"Saved {EMB_OUT} ({emb_mb:.1f} MB)")
    print(f"Saved {PKL_OUT} ({pkl_mb:.1f} MB)")
    print(f"Done. {len(df):,} movies | {elapsed:.1f}s elapsed")


if __name__ == "__main__":
    main()
