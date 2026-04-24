---
name: movie-recommendation-engine
description: 'Build a 4-hour school project movie recommender. Use when you need a working-first Streamlit app with local tmdb_5000_movies.csv, hybrid scoring, top-5 recommendations, and short why explanations.'
argument-hint: 'Describe your deadline, dataset status, and desired recommendation style.'
user-invocable: true
disable-model-invocation: false
---

# School Film Recommender (4-Hour Build)

## What This Skill Produces

This skill drives a strict workflow to produce a working, demo-ready AI-powered film recommendation project with:
- Streamlit UI
- Kaggle dataset file `movies_metadata.csv` from The Movies Dataset
- Hybrid recommendation logic (content similarity + popularity weighting)
- Top 5 recommendations
- Two-bullet why explanations for each recommendation
- Working-first implementation and validation

## When To Use

Use this when:
- You have a school deadline in a few hours
- Grading priority is functionality over polish
- You need one strong Kaggle file with minimal setup
- You need simple, robust architecture decisions quickly
- You want minimal questioning and maximum execution speed

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
- Required output (`top 5` and two-bullet why explanations)

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

### Step 3: Build Plan (Time-Boxed)

Create a strict plan that fits the remaining time.

Default 4-hour split:
1. Data loading and cleaning: 35 minutes
2. Recommender logic (hybrid scoring): 70 minutes
3. Streamlit UI: 55 minutes
4. Explanation generation and ranking polish: 25 minutes
5. Testing and bug fixing: 35 minutes
6. Demo script/readme notes: 20 minutes

If time left is less than 4 hours, proportionally shrink UI and polish first, never core functionality.

Completion check:
- Every block has a concrete output artifact.

### Step 4: Implementation Defaults

Use these default choices unless user overrides:
- Parse `genres` and `overview` text fields into normalized tokens.
- Build content vectors (TF-IDF or CountVectorizer) from combined token text.
- Compute content similarity from user seed movies.
- Add popularity prior from `popularity`, `vote_average`, and `vote_count` with conservative normalization.
- Hybrid score formula:
  - `score = 0.75 * content_score + 0.25 * popularity_score`
- Deduplicate and filter out seed movies from results.
- Return top 5 recommendations.

Completion check:
- Repeated runs with same input produce stable recommendations.

### Step 5: Why Explanations

For each recommendation, generate one short explanation using available signals:
- Shared genres
- Shared themes from overview text
- Strong audience signal (rating/popularity)

Explanation format:
- Exactly two bullets:
  - Similarity reason
  - Popularity/quality reason

Completion check:
- Each of top 5 has a distinct and credible why statement.

### Step 6: Working Validation

Run these tests before calling complete:
1. User enters valid 3 to 5 movie titles -> app returns 5 recommendations.
2. Unknown title handling -> app gives clear retry guidance.
3. Duplicate input titles -> app handles gracefully.
4. Missing fields in dataset -> app still runs with fallback logic.
5. Empty results edge case -> app returns a helpful fallback list.

Completion check:
- All 5 checks pass.

## Decision Branches

- If `movies_metadata.csv` is missing:
  - Ask for file path once.
  - If unavailable, switch to `tmdb_5000_movies.csv` and clearly disclose the fallback.

- If parsing is too slow or brittle:
  - Reduce features to `genres + keywords + overview`.
  - Keep hybrid weighting and top-5 output unchanged.

- If recommendation quality is weak:
  - Increase content weight to 0.8 and reduce popularity weight to 0.2.
  - Re-test explanations.

## Done Criteria

Declare completion only when all are true:
- App launches locally and accepts user movie seeds.
- Returns top 5 recommendations with why explanations.
- Handles common input and data edge cases.
- Logic and UI are understandable in a short live demo.

## Response Style

- Keep responses concise and execution-focused.
- Prefer concrete actions over theory.
- Surface blockers immediately with one proposed workaround.
- Do not over-question when constraints are already clear.
