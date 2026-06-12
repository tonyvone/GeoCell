# GeoCell: Funding Readiness & Build Roadmap

*An honest assessment: what we can prove today, what a serious investor
will actually require, and what to build to close the gap.*

---

## 1. The blunt truth about what we have

Everything to date **de-risks the technology thesis**. None of it, by
itself, gets a serious check written. Investors fund *evidence of a market
pulling the product*, and so far every number is from **synthetic data we
authored**. That is necessary and good — it proves the mechanism works —
but it is the "lab" stage, not the "traction" stage.

So the framing for this document: separate **tech-risk** (mostly retired)
from **market-risk** (untouched), and aim every next build at market-risk.

---

## 2. Results we already have (tech de-risking — credible, reproducible)

All reproducible in `evals/` against a real model with cached calls.

| Claim | Evidence | Strength |
|---|---|---|
| Beats the LLM on grounded Q&A | GeoOracle **100/100** vs LLM **80/100** | strong, synthetic |
| Cuts model calls | **−69%** on the same workload | strong, synthetic |
| Catches contradictions both LLM and vector DB miss | **10/10** traps vs **0/10** baseline | strong, synthetic |
| Hallucination quarantine works, no judge model | **5/5** fabricated answers caught | strong |
| Every served answer is cited | **100%** citation accuracy | strong |
| Coding repeat-fix deflection | **52%** of near-repeats fixed, 0 LLM | moderate, synthetic |
| Scales on CPU | recall p50 **153 ms @10k, 262 ms @50k** | moderate (prototype) |
| Poisoning-resistant | truth survives **500× low-authority flood** | strong, structural |
| Works on real text | PubMed abstracts: citations + trust ordering hold | **partial — real data** |
| Energy per deflected query | **~7,000–12,000× less** than the LLM call | strong (modeled) |

**The one real-data signal** (PubMed/ALS) is also where we found the honest
weakness: the epistemic layer works on real prose, but the rule-based
entity/claim extractor and the hash encoder degrade — and a controlled seam
swap proves the bottleneck is the pluggable NLP front-end, not the core.

---

## 3. What a serious investor will require (market de-risking — the gap)

The fundable milestones, in order of how much they move a term sheet:

### Tier 1 — Pre-seed / "this is real" (the immediate goal)

1. **One real customer corpus, reproduced by the customer.**
   Point GeoCell at a design partner's actual documents + query logs and
   produce: deflection %, served precision, and a list of **real
   contradictions caught that the customer confirms are genuine**. One
   slide of "we found 7 stale/conflicting policies in your own knowledge
   base that your LLM was answering wrong" is worth more than every
   synthetic table combined.
2. **A dollar figure on that corpus.** "At your query volume and your model
   pricing, this is $X/yr saved at Y% deflection" — computed from their
   numbers, not ours.
3. **The real-embedding result.** Show deflection *holds on paraphrase*
   (not just lexical matches) once a learned encoder replaces the hash
   encoder — proving the abstention bound we hit is an encoder limit, not a
   ceiling. Target: **≥80% deflection at ≥99% served precision** on a
   paraphrase-heavy set.
4. **2–3 signed design-partner LOIs.** Intent, not revenue, is enough at
   this stage — but it must be written.

### Tier 2 — Seed / "this works in production"

5. **A live deployment with sustained metrics over 8–12 weeks**: deflection
   and precision stable as the corpus churns; contradictions surfaced and
   resolved without manual intervention ≥90% of the time.
6. **Head-to-head vs the real alternatives** on the customer's data:
   GeoCell vs vector-DB RAG vs a temporal-KG (Zep) vs an LLM-judge — on
   precision/recall of contradiction detection, cost, and latency.
7. **An independent or instrumented energy/cost measurement** (not the
   top-down model) on production-like traffic.

### Tier 3 — Series A

8. Multiple paying customers, net revenue retention, an integration
   ecosystem (LangChain/LlamaIndex/MCP), and a measured productivity claim
   (e.g. "2× faster bug resolution in repo X").

**Where we are: end of lab stage, start of Tier 1.** The fastest path to a
check is items 1–3, and item 1 is gated by a design partner, not by code.

---

## 4. The specific numbers that would seal it

A fundable one-pager would read approximately:

- On **[Customer]'s** own 50k-document corpus and 30 days of real queries:
  **≥60% of queries answered without the LLM at ≥98% precision**, every
  answer cited.
- **≥85% of naturally-occurring contradictions** (confirmed by the
  customer's SMEs) detected automatically; **zero** of those caught by
  their existing vector-DB RAG.
- **$X00k/yr** projected inference saving at their volume + pricing.
- **Sub-300 ms** p95 served latency on commodity CPU, no GPU.
- Reproduced by the customer's own engineers from our package.

Hit those four and the synthetic results become the supporting appendix,
not the headline.

---

## 5. What I would build next (prioritized by fundability-per-effort)

### P0 — The fundraising instrument: **shadow-mode replay harness**
Point GeoCell at a customer's exported corpus + a sample of their real
query logs; run entirely offline, in read-only "shadow" mode alongside
their current system; emit a report: deflection %, served precision vs
their logged answers, **list of contradictions found with citations**, and
projected $ saved. This is the single highest-leverage thing to build — it
*manufactures the Tier-1 proof on the customer's own data with zero
production risk*, which is exactly what unlocks the design-partner
conversation. Build it so a partner can run it in an afternoon.

### P0 — **Real-embedding + domain-NER encoder** (closes the proven gap)
We *proved* the bottleneck is the NLP front-end via the seam experiment.
Wire a learned sentence embedding into `GeoCellField(encoder=…)` and a real
entity/claim extractor into `ingest(subject=…)`, then re-run Tests 2/3.
Expected: deflection on paraphrase jumps from "defer to LLM" to "served,"
and PubMed contradiction detection fires without the manual hint. (Blocked
in the current sandbox by no model download — needs an environment with
model access; this is a half-day once unblocked.)

### P1 — **Drop-in integrations** (distribution/adoption)
An MCP server and a LangChain/LlamaIndex retriever wrapper, so GeoOracle
slots into an existing agent in minutes. Adoption friction is a real
fund-or-not factor; "pip install, point at your docs, done" is the demo.

### P1 — **Calibration & eval dashboard**
Turn the abstention margin into a *calibrated* confidence (reliability
curve, precision-at-served vs deflection trade-off the operator can dial),
and a drift monitor (deflection/precision over time as the corpus churns).
This is what makes Tier-2 "sustained metrics" measurable and is itself a
selling point: an operator-tunable precision/cost knob.

### P2 — **Incremental + persistent backend at scale**
Move the inverted index and field to an embedded store (SQLite/LMDB) for
restart-survivable, larger-than-RAM operation; native/sparse inner loops to
push recall well under 50 ms at 1M cells. Needed for Tier-2 scale claims;
not needed to get the first check.

### P2 — **Multi-instance memory sync** (emergent capability)
Conflict-free merge of two GeoCell fields (CRDT-style on the cell ledger +
authority-weighted contradiction resolution on merge). Enables team/agent
shared memory and is a genuinely novel capability story for the A round.

---

## 6. Honest risks a diligent investor will probe

- **"Embeddings + a re-ranker already do retrieval."** True for recall; our
  wedge is the *belief layer* (contradiction, supersession, trust,
  abstention, audit) on top, which we've shown works on any encoder. We
  must demo that on real data, not assert it.
- **"Is the contradiction win real or a synthetic artifact?"** This is the
  crux. The PubMed probe is the start of the answer; a customer-confirmed
  contradiction list is the finish.
- **"Why won't OpenAI/a vector-DB vendor add this?"** Defensibility is the
  deterministic, auditable epistemology and the abstention calibration, not
  the storage. Honest answer: partly speed-to-market and focus.
- **Prototype maturity.** Pure-Python, single-node, rule-based extractors.
  Fundable as "we know exactly what to harden and why," not as "production."

---

## 7. Bottom line

We have retired most of the *technology* risk and localized the one real
weakness to a pluggable component. We have touched *none* of the *market*
risk. The next dollar of effort should go to the **shadow-mode harness** and
the **real-embedding swap**, because together they convert "great on our
benchmarks" into "great on your data, reproduced by you" — which is the
only result that actually gets funded.
