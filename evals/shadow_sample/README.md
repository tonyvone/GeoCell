# Shadow-Mode Replay — sample dataset

A tiny, self-contained example of the **shadow-mode replay harness**: the
go-to-market instrument that runs GeoCell over a customer's exported corpus
and a sample of their real query log, entirely **offline, read-only, and
with zero LLM calls**, and reports what it would deflect, where it agrees
with their current system, and — the money slide — **which answers their
system is giving from stale or contradicted data**.

## Run it

```bash
python -m geocell.shadow evals/shadow_sample/corpus.jsonl \
                         evals/shadow_sample/queries.jsonl \
                         --usd-per-query 0.01 --out report.json
```

## Input format (JSON Lines — trivial to export)

`corpus.jsonl` — one document per line:
```json
{"id":"...","source":"HR Policy Memo (CHRO, 2026)","date":"2026-01-15",
 "authority":0.95,"confidence":0.95,"text":"…","subject":"(optional NER hint)"}
```

`queries.jsonl` — one query per line; `answer` is the customer's logged
answer from their current system (optional, enables agreement scoring):
```json
{"query":"How many vacation days do full-time employees get?",
 "answer":"Full-time employees get 18 vacation days per year."}
```

## What this sample demonstrates

The corpus plants three **revisions** (a newer, higher-authority source
supersedes an older one) and the query log carries the **stale** answers a
current system would still be giving:

| Query | Incumbent (stale) | GeoCell finds |
|---|---|---|
| Vacation days | 18 | **22** (2026 CHRO memo) |
| Remote days/week | 2 | **3** (revised policy) |
| Project Orion budget | $2M | **$2.7M** (board update) |

On this sample GeoCell **deflects 80%** of queries, **agrees with the
incumbent on 5/8**, flags **3 disagreements** (the stale answers above,
each with a citation and belief status), surfaces **3 corpus
contradictions**, and correctly **escalates** the one genuinely
out-of-domain query ("generative AI policy" — absent from the corpus) and
one near-tie rather than guessing.

## Reading the report

- **deflection_rate** — fraction answerable from memory with no model call.
- **agreement_with_incumbent** — of deflected queries with a logged answer,
  how often GeoCell matches (number-aware).
- **disagreements** — the review list: GeoCell's answer + citation +
  belief status vs the incumbent's. These are stale/contradicted answers
  to investigate (the "we found N issues in your own data" output).
- **corpus_contradictions** — conflicting facts found in the corpus itself.
- **projected savings** — for the replayed sample; multiply by
  `annual_volume / sample_size` for the annual figure, using the
  customer's own `--usd-per-query` / `--wh-per-query`.

Everything is deterministic and reproducible: no LLM, no randomness, safe
to run on sensitive data in the customer's own environment.
