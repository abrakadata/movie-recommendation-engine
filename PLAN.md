# Build Plan — Movie Recommendation Engine

Total estimated time: 4 hours. Each step has a concrete output and a pass/fail check before moving on.

---

## Step 1 — Environment & Data (35 min)

**Goal:** Confirm the dataset loads and the right columns exist.

### Tasks
1. Create `app.py` with only imports and `DATA_PATH = "movies_metadata.csv"` at the top.
2. Write `load_data(path)` — reads CSV, selects columns, drops rows with empty `overview` or `genres`, returns a clean DataFrame.
3. Columns to keep: `title`, `genres`, `overview`, `popularity`, `vote_average`, `vote_count`.
4. Parse `genres` column: it arrives as a JSON string like `[{"id": 18, "name": "Drama"}]` — extract genre names into a plain Python list.

### Check before moving on
- Run `python app.py` in a `__main__` block, print `df.shape` and `df.head(2)`.
- Shape should be roughly `(4800, 6)` after drops.
- `genres` column should contain lists like `["Drama", "Romance"]`, not raw JSON strings.

---

## Step 2 — Content Vectors (40 min)

**Goal:** Build a semantic embedding matrix from movie text and compute similarity to seed movies.

### Tasks
1. Write `build_feature_text(df)` — concatenates genre names + overview into one string per movie.
   - Example: `"Drama Romance A young woman falls in love..."`.
2. Write `build_embedding_matrix(texts)` — encodes texts with `all-MiniLM-L6-v2` (sentence-transformers), returns a `(n, 384)` numpy array. Decorated with `@st.cache_data`.
3. Write `get_content_scores(seed_indices, matrix)` — computes cosine similarity between seed rows and all rows, averages across seeds, returns a 1D array of scores.

### Check before moving on
- Pick two known movies, get their row indices, call `get_content_scores`.
- Top 10 results should include movies with similar genres or themes — verify by eye.
- No NaN values in the score array.

---

## Step 3 — Popularity Score (20 min)

**Goal:** Add a normalized popularity prior so well-known films get a small boost.

### Tasks
1. Write `build_popularity_scores(df)` — combine `vote_average` and `vote_count` into a weighted score using the TMDB Bayesian formula or a simpler min-max normalization. Normalize to [0, 1].
   - Simple version: `score = (vote_average / 10) * log1p(vote_count) / max(log1p(vote_count))`.
2. Store result as a column `pop_score` on the DataFrame (done once at load time).

### Check before moving on
- Print the top 10 movies by `pop_score` — they should be recognizable blockbusters or classics.
- No values outside [0, 1].

---

## Step 4 — Hybrid Scoring & Top-5 Output (30 min)

**Goal:** Combine content and popularity scores, filter seeds, return top 5.

### Tasks
1. Write `recommend(seed_titles, df, matrix, n=5)`:
   - Fuzzy-match each seed title to a row (case-insensitive `str.contains`; take first match).
   - Collect seed indices; skip unmatched titles with a warning.
   - Call `get_content_scores` to get content array.
   - Hybrid: `score = 0.75 * content_score + 0.25 * pop_score`.
   - Zero out seed movie rows to exclude them from results.
   - Return top `n` rows as a DataFrame with columns: `title`, `genres`, `vote_average`, `score`.
2. Handle edge case: if fewer than 1 seed matches, return empty DataFrame.

### Check before moving on
- Call `recommend(["The Dark Knight", "Inception"], df, emb)`.
- Results should be 5 action/thriller films — none of the seeds should appear.
- Call with a typo like `"Incpetion"` — should still match (via `str.contains`) or gracefully skip.
- Call with all unrecognized titles — should return empty DataFrame without crashing.

---

## Step 5 — Why Explanations (20 min)

**Goal:** Generate a two-bullet explanation for each of the 5 results.

### Tasks
1. Write `explain(row, seed_genres)` — returns a two-item list of strings:
   - Bullet 1 (similarity): shared genre names between `row.genres` and the union of seed genres.
     - If no shared genres: fall back to `"Similar themes and storyline"`.
   - Bullet 2 (quality): `f"Rated {row.vote_average:.1f}/10 by audiences"` or `"Popular with audiences"` if vote_average is missing.
2. Call `explain` for each of the 5 results and attach to the output.

### Check before moving on
- Every row in the top-5 output has exactly two non-empty explanation strings.
- No crashes on rows with empty genre lists or missing vote_average.

---

## Step 6 — Streamlit UI (45 min)

**Goal:** A single-page app that takes movie titles and shows results.

### Tasks
1. Add `@st.cache_data` to `load_data` and `build_embedding_matrix` — called once at startup.
2. UI layout (top to bottom):
   - Title: `"Movie Recommender"`
   - Text input: `"Enter 3–5 movies you like (comma-separated)"`
   - Button: `"Find Movies"`
   - On click: parse input, call `recommend`, display results.
3. Results display: for each of the 5 recommendations show:
   - Movie title (bold)
   - Genres as a comma-separated string
   - Rating
   - Two explanation bullets
4. Error states:
   - No input: show `"Please enter at least one movie title."`
   - No matches found: show `"None of your titles were found. Try checking spelling."`
   - Fewer than 5 results: show however many exist with a note.

### Check before moving on
- Launch with `streamlit run app.py`.
- Enter `"The Dark Knight, Inception, Interstellar"` → 5 results appear with explanations.
- Clear input and click button → error message appears, no crash.
- Enter a completely unknown title → graceful message, no crash.

---

## Step 7 — Edge Case Hardening (20 min)

**Goal:** Close the five validation cases from the skill spec.

| Case | Expected behavior |
|---|---|
| 3–5 valid titles | Returns 5 recommendations |
| Unknown title | Clear retry message |
| Duplicate titles | Treated as one seed |
| Missing dataset fields | Fallback logic, no crash |
| Empty results | Helpful fallback message |

### Tasks
- Deduplicate seed list before matching.
- If `vote_average` or `vote_count` is NaN, fill with column median at load time.
- If `overview` is empty after stripping, fill with empty string (TF-IDF handles it).
- If the result set is empty, show the top 5 by `pop_score` as a fallback with a note: `"We couldn't match your titles. Here are some well-rated films."`.

### Check before moving on
- Manually trigger each of the five cases and confirm the expected behavior.

---

## Step 8 — README & Demo Script (10 min)

**Goal:** Enough documentation to run the app and explain it in 2 minutes.

### Tasks
1. Write a `README.md` with:
   - One-sentence description.
   - Install command: `pip install streamlit pandas scikit-learn`.
   - Run command: `streamlit run app.py`.
   - Dataset: place `movies_metadata.csv` in the same folder.
2. Mentally rehearse the demo: input 3 movies, show results, explain the hybrid scoring in one sentence.

---

## Time Budget Summary

| Step | Task | Time |
|---|---|---|
| 1 | Data loading | 35 min |
| 2 | Content vectors | 40 min |
| 3 | Popularity score | 20 min |
| 4 | Hybrid scoring | 30 min |
| 5 | Explanations | 20 min |
| 6 | Streamlit UI | 45 min |
| 7 | Edge case hardening | 20 min |
| 8 | README & demo prep | 10 min |
| **Total** | | **3h 40min** |

20-minute buffer for unexpected issues.

---

## Step 9 — Embedding Progress Bar ✅

**Goal:** Show a Streamlit progress bar while embeddings are being built so the user knows the app is working.

**Implemented:** Batched encoding with `st.progress`, disk cache at `embeddings_cache.npy`, `st.session_state` prevents recomputation on reruns. Progress bar only shows on cold start.

---

## Step 10 — Table Display for Recommendations ✅

**Goal:** Replace the card-style results display with a compact, scannable table.

**Implemented:** `st.dataframe` with `column_config` — Title, Genres, Rating (ProgressColumn), Why columns. `hide_index=True`, `use_container_width=True`.

---

## Step 11 — UI Polish ✅

**Goal:** Sleek, professional UI.

**Implemented:**
- Blue button (`#3b82f6`) with hover state (`#2563eb`)
- Non-collapsible sidebar with Recommendation Priority radio selector (3 weight presets)
- Sidebar header in Bootstrap green (`#198754`) at `2rem` bold
- Restart button replaces Find Movies after results are shown
- Title updated to "Movie Recommendation Engine"

---

## Step 12 — Dataset Cleanup ✅

**Goal:** Remove low-quality rows that degrade recommendation quality.

**Implemented:** Deleted 28 movies with empty genre lists (obscure/foreign/indie titles with no genre metadata). Dataset trimmed from 4,803 → 4,775 movies.

---
