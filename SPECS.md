# Architecture & Design Decisions

Non-obvious choices in the v2 codebase — the parts that can look arbitrary until you know what constraint they are serving.

---

## Embedding Pipeline

### Why embeddings are precomputed offline instead of inside Streamlit

The MPST dataset has 14,828 rows. Encoding all plot synopses on app startup would turn every cold start into a multi-second blocking operation and mix one-time preprocessing concerns with UI logic. `precompute.py` moves that work out of the request path: it cleans the dataset, computes embeddings once, joins popularity metadata, and writes the final runtime artifacts to disk.

At runtime, the app only loads `movies.pkl` and `embeddings.npy`.

### Why the final runtime dataset is `movies.pkl`

The app needs more than the raw MPST rows: it also needs the derived `pop_score` used for hybrid ranking. Rather than re-run the TMDB join and score computation on every page load, `precompute.py` writes the cleaned MPST dataframe plus the joined popularity fields into `movies.pkl`.

The two CSV files remain raw source inputs. No extra intermediate joined CSV is needed.

### Why `all-MiniLM-L6-v2` specifically

It produces 384-dimensional embeddings, which keeps the matrix small and cosine similarity fast while still performing well on plot-summary text. It is the practical model choice for a local school project: good semantic quality, CPU-friendly, and no GPU dependency.

### Why runtime loading uses `@st.cache_resource` and `@st.cache_data`

In v2, the expensive work is no longer embedding generation; it is just loading a model and reading precomputed artifacts. That maps cleanly to Streamlit's standard cache model:

- `load_model()` uses `@st.cache_resource` because the transformer model is a long-lived object.
- `load_data()` uses `@st.cache_data` because the dataframe and embedding matrix are pure loaded data.

This is simpler and more reliable than the v1 `st.session_state` embedding-build flow.

---

## Scoring

### Why v2 keeps hybrid scoring even though MPST has no popularity fields

MPST has the stronger text fields (`plot_synopsis`, `tags`) but no vote-based popularity metadata. Removing popularity entirely would make the recommender overly sensitive to semantic similarity alone and lose the useful bias toward broadly successful films. Instead, `precompute.py` joins `movies_metadata.csv` by normalized title and computes `pop_score` for matched rows.

Unmatched rows are still valid recommendations; they just rank on content similarity only.

### Why `log1p(vote_count)` is still used for popularity

Vote counts are extremely skewed. Raw counts would let major blockbusters dominate ranking even when the content match is weak. Using

`pop_score = (vote_average / 10) * log1p(vote_count) / max(log1p(vote_count))`

compresses the vote-count range so popularity acts as a prior rather than an override.

### Why unmatched TMDB rows use median rating and zero vote count

The title join is imperfect. Some MPST titles will not match a TMDB row. For those cases, filling `vote_average` with the dataset median avoids extreme synthetic ratings, while `vote_count = 0` ensures the final `pop_score` becomes `0.0` after the log-scaled formula. That keeps unmatched rows neutral instead of artificially boosted or suppressed.

### Why hybrid scores are blended at query time, not precomputed

`pop_score` is static per movie, but `content_scores` depend on the user's current query or selected seed movie. The app therefore computes:

`hybrid = similarity_weight * content_scores + (1 - similarity_weight) * df["pop_score"].values`

at interaction time. This keeps the sidebar `similarity_weight` slider live without invalidating the cached embeddings or the runtime dataframe.

### Why the min-similarity threshold is applied to content similarity, not the hybrid score

The threshold is intended to guard semantic relevance. If it were applied to the hybrid score, a popular but weakly related movie could survive the cutoff. Applying the threshold to `content_scores` preserves the meaning of the filter: below-threshold items are not similar enough, regardless of popularity.

### Why the selected movie is excluded by zeroing its score

In movie-similarity mode, the seed movie should never recommend itself. Zeroing its hybrid score preserves array/DataFrame index alignment and avoids a filtering step that would require remapping row indices after ranking.

---

## Navigation & UI

### Why mode selection uses `st.navigation` instead of a sidebar radio

The two discovery modes now have different interaction patterns: free-text search has a form-like input flow, while movie similarity has title selection plus a seed-movie summary. Treating them as separate pages makes each page simpler, keeps the entry point clean, and matches the user's mental model better than a single page with conditional branches.

`app.py` is intentionally minimal:

```python
pages = st.navigation([
	st.Page("pages/free-search-page.py", title="Free-text search", icon="🔍"),
	st.Page("pages/similarity-page.py", title="Movie similarity", icon="🎬"),
])
pages.run()
```

### Why `app.py` contains only navigation

Once the app is page-based, there is no reason for the entry point to own business logic. Keeping `app.py` as navigation-only reduces coupling and makes the page files the clear owners of each mode's UI and interaction flow.

### Why results stay in a table instead of cards

The core task is comparison, not browsing. A table makes it easier to scan multiple candidates quickly, compare similarity values side by side, and fit more results on screen without vertical sprawl. It also matches Streamlit's strengths better than a card-heavy layout.

The v2 results table uses:

- `Title`
- `Tags`
- `Match`
- `Synopsis`

instead of the v1 `Title / Genre / Rating` structure.

### Why the `Match` column is text with emoji instead of a progress bar

The match value is explanatory, not quantitative analysis. A compact string such as `🟢 87%` communicates the score and its rough quality band while keeping the column narrow. A progress bar would consume more width and add visual weight without improving decision quality.

### Why movie similarity mode includes a seed-movie info box

Once the user selects a movie, the app should anchor the recommendation context explicitly. Showing the selected title, tags, and a short synopsis excerpt above the results confirms which row was used as the seed and reduces ambiguity for common or similar titles.

---

## Filtering & Matching

### Why the tag filter is shared across both pages

The tag filter is a ranking constraint, not a page-specific interaction. Sharing it across both modes keeps the search mental model consistent: regardless of how candidates are discovered, the same semantic tag constraints can narrow the results.

### Why tag filtering uses "match any selected tag"

Requiring all selected tags would become too restrictive very quickly on a sparse semantic-tag dataset. "Any selected tag" is the more forgiving default and works better with discovery workflows where the user is trying to steer results rather than specify a strict boolean query.

### Why movie similarity uses a `selectbox` instead of title matching heuristics

In v1, a free-text title input forced fuzzy or substring matching concerns. In v2, movie-similarity mode uses a `selectbox` over known titles, which removes ambiguity and avoids the need for regex-safe substring matching logic in the UI path.

---

## Data Cleaning

### Why rows with missing `plot_synopsis` or `title` are dropped

These fields are essential to both modes. `plot_synopsis` is the primary embedding source, and `title` is required for display, seed selection, and the TMDB title join. Rows missing either field cannot participate meaningfully in the app and are removed during preprocessing.

### Why deduplication uses `imdb_id`

Title-based deduplication is unsafe because remakes, alternate releases, and reused names exist. `imdb_id` is the stable movie identifier in MPST, so it is the correct deduplication key.

### Why titles are normalized for the TMDB join

The popularity merge is based on title text, which is sensitive to case and surrounding whitespace. Lowercasing and stripping titles on both sides improves join recall at almost no cost and keeps the implementation simple enough for the project scope.

### Why whitespace is stripped from `title` and `tags`

This prevents avoidable display noise and inconsistent matching behavior. It is especially useful for the sidebar tag filter and for keeping the final table output clean.

---

## Performance

### Why the architecture is still local-first

Even with 14,828 rows, the workload remains small enough for a local laptop once embeddings are precomputed. The embedding matrix is about 21 MB in float32, model load is cached, and cosine similarity over the full matrix is fast on CPU. This keeps the app simple and deploy-free for a school project.

### Why there is no external API lookup at runtime

The app already has the data it needs once preprocessing is complete. Calling TMDB, OMDb, or another service during user interactions would add latency, API key handling, and failure modes that do not improve the core recommendation behavior.

---

## Project Scope Constraints

These are deliberate simplifications, not omissions.

- **No classes** — the code stays function-oriented for readability.
- **No test framework** — validation is lightweight and manual.
- **No config files** — paths and constants stay inline unless the project grows enough to justify extraction.
- **No LLM-generated explanations** — the recommender is embedding-based and deterministic.
- **Page split instead of over-abstraction** — the app is split into a minimal entry point plus two pages, but still avoids framework-heavy structure.

---

## Out of Scope

- User accounts or saved history
- Persistent ratings or a feedback loop
- Poster images or external API calls at runtime
- Cloud deployment concerns
- Cross-encoder re-ranking
- More advanced record linkage for TMDB/MPST title matching
