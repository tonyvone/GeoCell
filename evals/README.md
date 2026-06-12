# GeoCell Evaluations

Two end-to-end evals against a **real LLM** (the `claude` CLI, haiku), with
all calls disk-cached for reproducibility (`evals/.llm_cache/`).

## Test 1 — Repeat Bug / Verified Fix (`test1_coding.py`)

A sample service with 56 seeded bugs across auth, API routing, type
errors, config, database, and dependency misuse, organized into sibling
groups. An LLM authors fixes for the train half; only fixes the real
pytest suite accepts are distilled into GeoCell as generalized regex
transformations. The near-repeat test half is fixed by GeoCell FIRST,
with no LLM call, escalating to the LLM only on a miss. A patch-safety
gate (AST + import/attribute validation) quarantines hallucinated code.

Run: `PYTHONPATH=evals python evals/test1_coding.py`

## Test 2 — Verified Briefing / Hallucination Quarantine (`test2_knowledge.py`)

215 synthetic enterprise documents (internal docs, meeting notes,
contracts/SOWs, customer records) with planted exact facts, paraphrases,
multi-document chains, and contradiction traps. 100 gold questions are
answered LLM-only (baseline) and via GeoOracle (GeoCell first, LLM
fallback). LLM answers that contradict trusted memory are quarantined.

Run: `PYTHONPATH=evals python evals/test2_knowledge.py`

Results are written to `evals/test1_results.json` / `evals/test2_results.json`.
