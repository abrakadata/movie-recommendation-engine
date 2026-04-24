# Project Constitution — Movie Recommender (First-Year AI School Project)

## Purpose

This document defines the rules and constraints that govern all implementation decisions. Its job is to prevent over-engineering, scope creep, and premature abstraction. If a decision is not covered here, default to the simplest option that works.

---

## Core Rules

### 1. Working beats elegant
The app must run and produce results. A working ugly solution ships. A clean broken one does not.

### 2. One file, then split
Start with a single Python script. Only split into multiple files if a single file becomes genuinely unreadable (>300 lines). No package structure, no `src/` layout, no `__init__.py`.

### 3. No unnecessary dependencies
Allowed libraries: `pandas`, `scikit-learn`, `streamlit`, `numpy`. Do not add anything not in this list unless it solves a problem you actually have. No LLM API calls, no embeddings libraries, no vector databases.

### 4. No configuration files
No `.env` files, no `config.yaml`, no `settings.py`. Constants go inline where they are used. Move them to the top of the file only if they appear more than once.

### 5. No classes unless unavoidable
Functions only. A class is allowed only if the data and the functions that use it are genuinely inseparable. A recommender that is three functions is better than a `Recommender` class with three methods.

### 6. No async, no caching layers, no queues
Pandas and scikit-learn are fast enough for a 45,000-row CSV. No `@st.cache_data` over-engineering — one `@st.cache_data` on the data loader is fine and sufficient.

### 7. Hardcode the dataset path
`DATA_PATH = "movies_metadata.csv"` at the top of the file. No file-picker UI, no path argument, no environment variable.

### 8. No test framework
Validation is done by running the app and checking outputs manually, plus two or three inline `assert` statements in a `__main__` block. No `pytest`, no `unittest`.

### 9. UI is one page
Single Streamlit page. No tabs, no sidebar navigation, no multi-page app. A text input, a button, and a results section. Done.

### 10. Explanations are string templates
Why-explanations are built with f-strings using genre lists and rating numbers pulled directly from the DataFrame. No LLM, no NLP pipeline, no summarizer.

---

## What "First-Year School Project" Means Here

| Allowed | Not Allowed |
|---|---|
| TF-IDF with cosine similarity | Neural embeddings, transformers |
| Pandas for all data work | Spark, Dask, Polars |
| Streamlit UI | Flask, FastAPI, React |
| Inline constants | Config files, env vars |
| Manual spot-check tests | pytest, mock, fixtures |
| One hybrid score formula | A/B tested weighting, ML-tuned weights |
| f-string explanations | GPT-generated summaries |
| `movies_metadata.csv` or `tmdb_5000_movies.csv` | API calls, scraping, databases |

---

## Decision Checklist (run this before adding anything)

1. Does this make the app work better for the demo? If no, skip it.
2. Does a simpler version exist? Use that instead.
3. Will a first-year student reading this understand it in 5 minutes? If no, simplify.
4. Does it require a new dependency? If yes, justify it or drop it.
5. Is it needed before the deadline? If no, skip it.
