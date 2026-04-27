# Implementation Plan — v2

Ordered checklist. Complete each step fully before moving to the next. Steps within a phase can be verified in isolation.

---

## Phase 1 — Precompute (`precompute.py`)

This script runs once from the terminal before launching the app. It produces the two runtime artifacts the app depends on.

- [ ] **1.1** Load `mpst_full_data.csv` with pandas
- [ ] **1.2** Drop rows where `plot_synopsis` or `title` is null
- [ ] **1.3** Deduplicate on `imdb_id`, keep first occurrence
- [ ] **1.4** Strip whitespace from `title` and `tags` columns
- [ ] **1.5** Load `movies_metadata.csv`, keep `title`, `vote_average`, `vote_count`
- [ ] **1.6** Normalize titles in both datasets (lowercase + strip) into temporary join-key columns
- [ ] **1.7** Left-join MPST df on normalized title → TMDB popularity fields
- [ ] **1.8** For unmatched rows: fill `vote_average` with TMDB median, fill `vote_count` with `0`
- [ ] **1.9** Compute `pop_score = (vote_average / 10) * log1p(vote_count) / max(log1p(vote_count))`
- [ ] **1.10** Drop the temporary join-key columns; keep only MPST columns + `pop_score`
- [ ] **1.11** Reset index
- [ ] **1.12** Load `SentenceTransformer("all-MiniLM-L6-v2")`
- [ ] **1.13** Encode `plot_synopsis` in batches of 64 with `show_progress_bar=True`
- [ ] **1.14** Save embeddings as `embeddings.npy` (float32, shape [N, 384])
- [ ] **1.15** Save cleaned dataframe as `movies.pkl`
- [ ] **1.16** Print: row count, both file sizes, time elapsed

**Verify:** Run `python precompute.py`. Confirm `movies.pkl` and `embeddings.npy` exist. Confirm `pop_score` column is present in the loaded pkl. Confirm no nulls in `title` or `plot_synopsis`.

---

## Phase 2 — Entry Point (`app.py`)

Replace all existing content with navigation-only.

- [ ] **2.1** Delete all existing logic from `app.py`
- [ ] **2.2** Add `st.navigation` with two pages:
  - `pages/free-search-page.py` — title `"Free-text search"`, icon `"🔍"`
  - `pages/similarity-page.py` — title `"Movie similarity"`, icon `"🎬"`
- [ ] **2.3** Call `pages.run()`

**Verify:** `streamlit run app.py` launches without error. Both page tabs appear in the top nav bar.

---

## Phase 3 — Shared Utilities (`utils.py`)

Both pages need the same model and data. Centralise loading here.

- [ ] **3.1** Create `utils.py` in project root
- [ ] **3.2** Implement `load_model()` with `@st.cache_resource` — returns `SentenceTransformer("all-MiniLM-L6-v2")`
- [ ] **3.3** Implement `load_data()` with `@st.cache_data` — loads `movies.pkl` (dataframe) and `embeddings.npy` (numpy array), returns both
- [ ] **3.4** Add guard: if either file is missing, call `st.error("Run python precompute.py first.")` and `st.stop()`
- [ ] **3.5** Implement `render_sidebar(df)` — renders all shared sidebar widgets and returns a dict:
  - `n_results` — slider, 3–15, default 5
  - `tag_filter` — multiselect, options from `sorted(set(tag.strip() for tags in df["tags"] for tag in tags.split(",")))`, default empty
  - `min_similarity` — slider, 0.0–1.0, default 0.0, step 0.05
  - `similarity_weight` — slider, 0.0–1.0, default 0.75, step 0.05, label `"Similarity weight (vs. popularity)"`
- [ ] **3.6** Implement `format_match(score)` — returns `"🟢 {n}%"` if ≥ 0.75, `"🟡 {n}%"` if ≥ 0.50, else `"🔴 {n}%"` where `n = round(score * 100)`
- [ ] **3.7** Implement `apply_filters(df, content_scores, tag_filter, min_similarity)` — zeros out scores for rows that fail the tag filter or fall below the similarity threshold; returns the modified score array
- [ ] **3.8** Implement `display_results(df, indices, scores)` — builds and renders the `st.dataframe` with columns: `Title`, `Tags`, `Match` (via `format_match`), `Synopsis` (first 200 chars + `"…"`)

**Verify:** Import `utils` in a scratch script; call `load_data()` and confirm shapes match expected row count and 384 dimensions.

---

## Phase 4 — Free-Text Search Page (`pages/free-search-page.py`)

- [ ] **4.1** Create `pages/` directory
- [ ] **4.2** Create `pages/free-search-page.py`
- [ ] **4.3** Import from `utils`: `load_model`, `load_data`, `render_sidebar`, `apply_filters`, `display_results`
- [ ] **4.4** Load model, df, and embeddings at module level (cached)
- [ ] **4.5** Call `render_sidebar(df)` and unpack the returned controls
- [ ] **4.6** Render `st.text_area` — label `"Describe the kind of movie you want to watch"`, placeholder `"A psychological thriller where nothing is what it seems"`, height 100
- [ ] **4.7** Render `st.button("Find Movies")`
- [ ] **4.8** On button click:
  - Validate: show `st.warning("Please enter a description.")` if blank and stop
  - Encode query: `query_vec = model.encode([query_text])`
  - Compute `content_scores = cosine_similarity(query_vec, embeddings)[0]`
  - Compute `hybrid = similarity_weight * content_scores + (1 - similarity_weight) * df["pop_score"].values`
  - Call `apply_filters` to zero out tag/threshold exclusions
  - Rank: `top_idx = hybrid.argsort()[::-1][:n_results]`
  - If all scores are zero after filtering: show `st.info("No movies matched your filters...")` and stop
  - Call `display_results(df, top_idx, content_scores[top_idx])`

**Verify:** Enter a description, click Find Movies. Table appears with correct columns. Tag filter and similarity threshold narrow results correctly.

---

## Phase 5 — Movie Similarity Page (`pages/similarity-page.py`)

- [ ] **5.1** Create `pages/similarity-page.py`
- [ ] **5.2** Import from `utils`: `load_model`, `load_data`, `render_sidebar`, `apply_filters`, `display_results`
- [ ] **5.3** Load model, df, and embeddings at module level (cached)
- [ ] **5.4** Call `render_sidebar(df)` and unpack the returned controls
- [ ] **5.5** Render `st.selectbox("Pick a movie you like", options=sorted(df["title"].tolist()))`
- [ ] **5.6** Look up selected title's index in df (exact match against sorted list — no fuzzy matching needed)
- [ ] **5.7** Render seed movie info box using `st.info()`:
  ```
  🎬  <Title>
  Tags: <tags>
  <First 200 characters of plot_synopsis>...
  ```
- [ ] **5.8** Compute:
  - `seed_vec = embeddings[idx].reshape(1, -1)`
  - `content_scores = cosine_similarity(seed_vec, embeddings)[0]`
  - `hybrid = similarity_weight * content_scores + (1 - similarity_weight) * df["pop_score"].values`
  - `hybrid[idx] = 0.0` — exclude seed from results
- [ ] **5.9** Call `apply_filters` to zero out tag/threshold exclusions
- [ ] **5.10** Rank: `top_idx = hybrid.argsort()[::-1][:n_results]`
- [ ] **5.11** If all scores are zero after filtering: show `st.info("No movies matched your filters...")`  and stop
- [ ] **5.12** Call `display_results(df, top_idx, content_scores[top_idx])`

**Verify:** Select a movie. Seed info box appears above table. Selected movie does not appear in results. Tag filter works. Changing the similarity weight slider reruns and re-ranks.

---

## Phase 6 — Smoke Test

Manual end-to-end checks before calling v2 done.

- [ ] **6.1** Cold start: delete `movies.pkl` and `embeddings.npy`, run `python precompute.py`, confirm both files regenerated
- [ ] **6.2** Missing file guard: rename `movies.pkl` temporarily, confirm `st.error` and `st.stop` fire on page load
- [ ] **6.3** Free-text search: empty submit shows warning; short description returns results; all Match values show emoji prefix
- [ ] **6.4** Free-text search: select 2–3 tags, confirm only tag-matching movies appear
- [ ] **6.5** Free-text search: raise min-similarity to 0.80, confirm row count drops
- [ ] **6.6** Movie similarity: selected movie is never in the results table
- [ ] **6.7** Movie similarity: seed info box shows correct title, tags, and synopsis excerpt
- [ ] **6.8** Both pages: changing `n_results` slider changes table row count
- [ ] **6.9** Both pages: changing similarity weight slider re-ranks results (high weight → top results have strong synopsis match; low weight → popular titles rank higher)
- [ ] **6.10** Navigate between the two pages; sidebar state persists across page switch

---

## Files Created / Modified

| File | Action |
|---|---|
| `precompute.py` | Create |
| `app.py` | Rewrite (navigation only) |
| `utils.py` | Create |
| `pages/free-search-page.py` | Create |
| `pages/similarity-page.py` | Create |
| `embeddings_cache.npy` | Delete (replaced by `embeddings.npy`) |
| `movies_metadata.csv` | Keep as raw input |
| `mpst_full_data.csv` | Keep as raw input |
