## Sudent: Abner Wilding, Class: CAI1001C-2263 | Artificial Intelligence Thinking

# Project: Movie Buddy Buddy
# An AI-Powered Movie Recommendation Engine

### A Streamlit movie recommendation engine with sentence-transformer embeddings, hybrid scoring, configurable priority preferences, table results display, and embedding progress bar.


## What This Skill Produces

A working, demo-ready AI-powered movie recommendation desktop app with:
- Streamlit UI with sidebar, table results, and Restart button
- TMDB dataset (`movies_metadata.csv`, ~4,775 movies)
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384-dim)
- Hybrid recommendation logic (content similarity + popularity weighting)
- Configurable weighting via sidebar (75/25, 25/75, 50/50)
- Top 5 recommendations in an interactive `st.dataframe` table
- Two-bullet why explanations per recommendation
- Embedding progress bar on first launch, `.npy` disk cache on subsequent runs

## When To Use

Use this when:
- You need a working recommendation engine quickly
- You want a polished Streamlit UI with minimal setup
- You have `movies_metadata.csv` (TMDB) available locally
- You want hybrid scoring with user-controllable weights

## Interaction Rules

1. Ask only deadline-critical questions.
2. Keep clarification to the minimum needed to avoid rework.
3. If the user provides enough constraints, stop asking and execute.
4. Prefer defaults that reduce risk and complexity.
5. Always explain tradeoffs in plain language.

## Workflow

### Step 1: Scope Lock (Fast)

Collect only these essentials:
- Deadline and grading priority
- Data source availability (`movies_metadata.csv` present or not)
- Interface (`Streamlit`)
- Required output (top 5 + why explanations)

If any item is missing, ask one concise question per missing item.

Completion check:
- Scope can be written in one paragraph with no contradictions.

### Step 2: Approach Synthesis

Generate exactly 3 implementation approaches that are:
- Simple
- Robust
- Elegant

For each approach provide:
- Core design
- Why it fits the constraints
- Main tradeoff

Then pick one recommended approach and justify it using:
- Deadline fit
- Failure risk
- Demo clarity

Completion check:
- User can choose an approach in under 2 minutes.

### Step 3: Build Plan

Create a strict plan that fits the available time of ~6 hours.

1. Data loading and cleaning: 45 minutes
2. Sentence-transformer embeddings + cosine similarity: 45 minutes
3. Popularity score (Bayesian normalization): 30 minutes
4. Hybrid scoring + top-5 output: 30 minutes
5. Why explanations: 44 minutes
6. Streamlit UI (sidebar, table, progress bar, Restart button): 120 minutes
7. Edge case hardening: 30 minutes
8. README and demo prep: 10 minutes
9. Security review and final polish: 15 minutes
10. Buffer for unexpected issues: 15 minutes

If time is short, shrink UI polish first — never cut Steps 1–4.

Completion check:
- Every block has a concrete output artifact.

### Step 4: Implementation Defaults

Use these default choices unless user overrides:
- Parse `genres` JSON strings into plain Python lists.
- Drop rows with empty `overview` or empty genre lists.
- Build feature text: `genres joined by space` + `overview`.
- Encode with `SentenceTransformer("all-MiniLM-L6-v2")` in batches of 64.
- Cache matrix to `embeddings_cache.npy`; load on subsequent runs.
- Use `st.session_state` to avoid recomputing on Streamlit reruns.
- Popularity score: `(vote_average / 10) * log1p(vote_count) / max(log1p(vote_count))`.
- Default hybrid formula: `score = 0.75 * content_score + 0.25 * pop_score`.
- Expose weighting via sidebar radio with three presets.
- Deduplicate seed titles; exclude seed movies from results.
- Return top 5 as `st.dataframe` with Title, Genres, Rating (ProgressColumn), Why.

Completion check:
- Repeated runs with same input produce stable recommendations.

### Step 5: Why Explanations

For each recommendation generate two bullets:
- Similarity: shared genre names between result and seed union; fallback to "Similar themes and storyline".
- Quality: `"Rated X.X/10"`if rating missing.

Display as a single joined string in the Why column (` · ` separator).

Completion check:
- Each of top 5 has a distinct and credible why statement.

### Step 6: Working Validation

Run these tests before calling complete:
1. Valid 3–5 movie titles → app returns 5 recommendations in table.
2. Unknown title → warning shown, remaining seeds still used.
3. Duplicate input titles → treated as one seed.
4. Missing dataset fields → fallback logic, no crash.
5. Empty results → top 5 by pop_score shown with explanatory note.
6. Restart button → clears results, returns to input state.

Completion check:
- All 6 checks pass.

## Decision Branches

- If `movies_metadata.csv` is missing: ask for file path once.
- If embedding build is slow: reduce batch size to 32 or use a smaller model.
- If recommendation quality is weak: increase content weight to 0.8.
- If a movie is not found: warn and skip — do not crash.

## Done Criteria

Declare completion only when all are true:
- App launches and accepts movie seeds.
- Returns top 5 in a table with why explanations.
- Sidebar weight selector works correctly.
- Handles common input and data edge cases.
- Restart button resets the app cleanly.
- Logic and UI are understandable in a short live demo.

## Response Style

- Keep responses concise and execution-focused.
- Prefer concrete actions over theory.
- Surface blockers immediately with one proposed workaround.
- Do not over-question when constraints are already clear.
