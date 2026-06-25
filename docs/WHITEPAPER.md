# GeoCell — Technical & Business Whitepaper

**A deterministic, geometry-native memory layer that makes LLM systems
cheaper, more accurate, and auditable.**

*Version 0.5 · all figures reproducible from `geocell/` and `evals/` ·
49 passing tests · CPU-only, no GPU, no trained weights in the core.*

---

## Executive summary

Large language models are deployed over knowledge that **changes** and at a
cost that **compounds**. Two failures dominate production: models cannot
distinguish a current fact from a superseded one, and every answer —
including trivially repeated ones — costs a full, energy-intensive
inference. The cost problem is worst exactly where the industry is heading:
long-context prompts and multi-step agents, where token cost scales
super-linearly.

**GeoCell** is a memory engine that sits in front of any LLM. It models
memory as a trust-weighted geometric field in which each fact carries
provenance and a belief status (active / contested / superseded /
retracted). It answers from memory whatever it can answer with high
confidence — instantly, for free, with a citation — escalates only novel
questions to the model, and **quarantines model answers that contradict
trusted memory**. The entire epistemic layer is deterministic arithmetic:
reproducible, auditable, and free of any secondary "judge" model.

**Measured results (reproducible, real model, cached calls):**

- **Knowledge Q&A:** GeoCell-fronted (GeoOracle) scores **100/100 vs 80/100**
  for the LLM alone, with **69% fewer model calls**, **100% of served
  answers cited**, and **10/10 contradiction traps caught vs 0/10**.
- **Coding:** **52%** of near-repeat bugs fixed with **no** model call;
  fix proposals **~1,100× faster** than the model; **5/5** hallucinated
  patches blocked.
- **Long-context / agentic (the high-leverage regime):** per grounded RAG
  query **~96% cost reduction**; a 30-step agent **~70%**, a 100-step agent
  **~90%**, because GeoCell turns the agent's cumulative token cost from
  **O(T²) to O(T)** in step count.
- **Efficiency physics:** a deflected query uses **~7,000–12,000× less
  energy** than the LLM call it replaces.
- **Robustness:** a trusted fact **survives a 500× low-authority poisoning
  flood**; the design weights by source authority, not vote count.
- **Footprint:** **96 bytes per memory** quantized (32× smaller than float
  vectors), served from CPU cache.

The technology risk is largely retired and the one real weakness is
localized to a **pluggable** NLP front-end (entity extraction, encoder).
The remaining work is market validation and standard production hardening,
both de-risked by a **shadow-mode** deployment pattern that proves value on
a customer's own data with zero production risk.

---

# PART I — TECHNICAL

## 1. The core idea

A database stores rows. A vector store ranks by similarity. Neither has a
native concept of *what is true now*, *what displaced it*, or *what is
disputed*. GeoCell's primitive is neither a row nor a token but a
**deformable geometric memory object** — a particle in a high-dimensional
field — carrying:

| facet | role |
|---|---|
| position (unit hypersphere) | where the memory sits in meaning-space |
| lexical anchor | the original encoding; the cell is tethered to it |
| provenance | source, date, confidence, **authority** |
| trinary relations | support / unknown / **contradiction** to neighbors |
| belief status | active / contested / superseded / retracted |
| propagated trust | a fixed-point score over the relation graph |
| ledger | append-only audit trail of everything that happened to it |

Evidence exerts force: corroboration attracts, contradiction repels, trust
flows. The result is memory as **physics with an epistemology**.

## 2. Architecture

### 2.1 Encoding (deterministic, no trained weights)

A sentence maps to a unit vector by hashing surface features (uni/bi/tri-
grams), **role-bound** features (subject and extracted relation triples
occupy their own coordinates), magnitude buckets for numeric claims, and a
negation axis. A conservative stemmer adds morphological co-features
(owns/owned/owner share a slot). The map is identical across runs and
machines — the geometry is stable and contains no model weights. **The
encoder is pluggable** (`GeoCellField(encoder=…)`): a learned embedding can
replace the hash encoder without touching anything above it.

### 2.2 Belief lifecycle (the differentiator)

When a dated claim conflicts with an older one on the same subject and
predicate, GeoCell does not overwrite. If the newcomer carries a revision
signal or strictly higher authority, it **supersedes** the older claim,
which is retained as cited history and down-weighted in recall but
recoverable by historical queries. Equal-strength conflicts leave both
**contested** and surfaced in every relevant answer. A derived hypothesis
that conflicts with an observation is **retracted**. Every transition is
written to the cell's ledger.

### 2.3 Trust propagation

Trust is the fixed point of a damped iteration over the relation graph:

```
support_i  = Σ_j w_ij·t_j / Σ_j w_ij          (flows along support edges)
pressure_i = max_j (t_j − t_i)                 (more-trusted opponents push down)
t_i ← clip(0.55·prior_i + 0.45·support_i − 0.30·pressure_i)
```

with caps on superseded/retracted cells. `why(id)` returns the exact
decomposition (prior, support flow, contradiction pressure) for any belief.
Because trust is driven by **source authority, not vote count**, a flood of
weak sources cannot outvote one trusted source — the structural defense
against write-poisoning.

### 2.4 Hybrid retrieval + confident abstention

Recall blends the geometric cosine with **BM25** over an inverted index, so
a rare, discriminating term ("vendor", "owned") outranks the common subject
every candidate repeats; energy then diffuses two hops along support edges
(spreading activation) so a query can surface facts it never lexically
mentions. Critically, recall exposes a **top-1-vs-top-2 margin**: the
serving layer answers only on a clear winner and **defers near-ties to the
LLM**. This selective-prediction property is why served accuracy is 100% —
the system never guesses.

### 2.5 Defeasible inference

Transitive rules over extracted relations (e.g. `part_of ∘ depends_on ⇒
depends_on`) generate hypothesis cells that live in the field like any
memory: they answer questions, carry a derivation chain to observed
sources, inherit discounted trust, and are retracted on contradiction with
observation — yielding multi-hop answers absent from any single document.

### 2.6 Serving path

- **Binary quantization:** each position/anchor is snapped to a 768-bit
  SimHash signature (**96 bytes**, 32× smaller); similarity becomes
  XOR+popcount with full-precision queries scored asymmetrically against
  quantized cells. 100k memories = ~9.6 MB index, served from CPU cache.
- **GeoOracle:** wraps any `llm(prompt)→str`. Resonant queries answered
  from memory (cited, with belief status); novel queries escalate and are
  absorbed; LLM answers conflicting with trusted memory are **quarantined**
  (trusted value + citation returned) with a full audit trail and **no
  secondary judge model**.

### 2.7 Scale engineering

Ingestion uses inverted-index candidate wiring (not all-pairs) → bounded
~5–7 ms/cell. Recall uses inverted-index candidate generation + cached
support adjacency + vectorized scoring over a bounded scope → **does not
grow with field size**. Measured on pathologically dense synthetic data
(88–220 edges/cell, far above natural text), pure-Python CPU prototype:

| Cells | Ingest | Recall p50 | Recall p95 |
|---|---|---|---|
| 10,000 | ~5 ms/cell | 153 ms | 210 ms |
| 50,000 | ~7 ms/cell | 262 ms | 302 ms |

## 3. Validation

All benchmarks are reproducible against a real model with cached calls
(`evals/`).

**Knowledge (100 questions, 215-doc corpus):** GeoOracle **100/100** vs LLM
**80/100**; **69%** calls avoided; **100%** citation accuracy; **10/10**
contradiction traps vs **0/10** baseline; **5/5** fabricated answers
quarantined. By type: exact recall 30→40, synthesis 4→20, near-repeat and
contradiction both fully handled (deferring ambiguous cases to the model).
The cascade is strictly better than either component alone.

**Coding (21 near-repeat bugs):** **52%** fixed with no model call; **5/5**
hallucinated patches blocked by an AST safety gate; 90% overall fix; a
multi-bug repo moved 33→44 passing tests with zero model calls.

**Real data (PubMed/ALS, 4 real abstracts 2011–2025):** the epistemic
layer works on real prose — per-answer citations to real papers, trust
ordered by journal authority + recency, fully deterministic. The honest
finding: the **rule-based entity extractor degrades** on natural prose
("repeat expansion" instead of "C9orf72"), and a controlled swap to an
extractor seam restores contradiction detection — isolating the bottleneck
to the pluggable front-end, not the core.

**Adversarial:** one high-authority truth survives a **500×** low-authority
contradiction flood; trust does not collapse; lies are superseded, not
adopted.

## 4. Honest technical limitations

1. The hash encoder is weak on deep semantic paraphrase; GeoCell handles
   this by deferring such queries (100% served precision, bounded
   deflection). Fixable via the encoder seam with real embeddings.
2. Rule-based entity/claim extraction degrades on natural prose; fixable
   via the extractor seam with a real NER.
3. Pure-Python, single-node prototype: memory layout, write-path latency,
   durability (WAL), deletion (GDPR), and concurrency are not yet
   production-grade (see `docs/PRODUCTION.md` for the gap analysis and plan).
4. Benchmarks are synthetic (realistic, but not a customer corpus); energy
   figures are transparent estimates, not audited measurements.

None of these are research risks; all are known engineering with clear
acceptance tests.

---

# PART II — EFFICIENCY ECONOMICS

## 5. Per-query

A GeoCell-served answer costs ~0.03–0.4 mWh of CPU vs ~0.24–0.34 Wh for an
average model query — **~7,000–12,000× less energy** per deflected query.
Net energy reduction across a served workload ≈ the deflection fraction
(CPU overhead is negligible): **~40% at conservative 0.40 deflection, ~69%
at measured deflection.**

## 6. Long-context & agentic — where the leverage is largest

Token cost compounds super-linearly in these regimes; GeoCell breaks the
compounding three ways (`evals/agentic_efficiency.py`, conservative —
baseline already uses prompt caching, win is on the uncacheable growing
context):

1. **RAG context compression.** Replacing k-chunk stuffing (20×400 = 8,800
   input tokens) with one verified, cited fact (~310): **~96% cost
   reduction per grounded query.**
2. **Agentic context growth — O(T²) → O(T).** A naive agent re-sends its
   whole growing transcript every step (quadratic cumulative tokens);
   GeoCell holds the memory and the agent keeps a bounded working window
   (linear). Reduction by run length:

   | Agent length | Token/cost reduction | Baseline:GeoCell input ratio |
   |---|---|---|
   | 10 steps | ~33% | 1.5× |
   | 30 steps | **~70%** | — |
   | 80 steps | — | **7.9×** |
   | 100 steps | **~90%** | — |

   The longer the agent, the larger the saving — the difference between
   long-horizon agents being economically prohibitive and viable.
3. **Deflection (additive).** Lookup-shaped agent steps serve from memory
   with no model call: at a conservative 20–40% rate, the 30-step reduction
   rises from 70% to **77–83%**.

**At scale:** 100,000 thirty-step agent runs/day, compression alone, saves
**~$28M/year and ~2.8 GWh/year** — before stacking RAG compression and
deflection.

## 7. Market-scale energy

Layering IEA data-center totals (485 TWh in 2025 → 950 TWh in 2030, AI
share 12%→40%) with inference ≈ 55% of AI compute and grounded/repeat ≈ 40%
of inference, the GeoCell-addressable envelope is **12.8 TWh/yr (2025)** and
**83.6 TWh/yr (2030)**. At measured deflection and 50% adoption that is
**~4.4 TWh/yr (2025) rising to ~28.8 TWh/yr (2030)** — 0.9–3.0% of *all*
data-center electricity, equivalent to the residential power of millions of
homes. (Transparent estimate; assumptions in `evals/energy_model.py`.)

---

# PART III — BUSINESS

## 8. The market and why now

Inference, not training, is now the dominant operational cost of AI
products, and it is growing fastest exactly where GeoCell helps most:
RAG assistants, support copilots, and multi-step agents. Simultaneously,
regulated buyers (finance, legal, healthcare) cannot ship confident-but-
wrong answers and increasingly require citations and audit trails. Existing
memory stacks — vector DBs, and even temporal knowledge graphs — either
ignore contradictions or call an LLM to detect them. GeoCell is the only
layer that does contradiction handling **deterministically and auditably**
while **cutting model spend**.

## 9. Use cases (ICP, in priority order)

1. **Regulated knowledge work** — finance/legal/healthcare assistants. The
   contradiction + citation + audit story is the wedge; cost is the bonus.
2. **High-volume AI products** — support and internal copilots, where a
   50–70% call deflection is a direct, large line-item saving.
3. **Agent platforms** — where the O(T²)→O(T) context economics make
   long-horizon agents viable, plus a hallucinated-action gate.

## 10. Competitive landscape

| | Raw LLM | Vector DB / RAG | Temporal KG (Zep/Graphiti) | **GeoCell** |
|---|---|---|---|---|
| Grounded recall | ✗ | ✓ | ✓ | ✓ |
| Handles contradictions | ✗ | ✗ | ✓ (LLM-based) | ✓ (deterministic) |
| Catches LLM hallucination | ✗ | ✗ | partial | ✓ |
| Cuts model calls | ✗ | partial | partial | ✓ (−69%) |
| Long-context/agent economics | ✗ | ✗ | ✗ | ✓ (O(T²)→O(T)) |
| CPU-only, no GPU | n/a | mostly | mostly | ✓ |
| Audit trail + citations | ✗ | source only | ✓ | ✓ (+ ledger, trust) |

**Defensibility:** the deterministic, auditable epistemology and the
calibrated abstention — not storage. The encoder/extractor are commodities
(and pluggable); the belief layer on top is the moat, and it works with any
encoder.

## 11. Positioning

> **The verified memory layer for AI** — the trust and cost-control tier
> between your agents and your LLM.

Three headlines: **Spend less** (−69% calls; O(T²)→O(T) on agents) ·
**Hallucinate less** (deterministic quarantine, no judge model) ·
**Prove it** (every answer cited, with belief status and a ledger).

## 12. Unit economics & business model

GeoCell's cost to serve is CPU-cents; the value delivered is a fraction of
the model spend it eliminates. Illustrative, at representative pricing:

- **Flat workload** (1M grounded queries/day, ~$0.005/query, 60% deflection):
  ~**$1.1M/yr** saved on that workload.
- **Agentic workload** (100k 30-step runs/day): ~**$28M/yr** saved
  (compression alone).

**Model:** usage-based, priced as a fraction (e.g. 20–30%) of the model
spend deflected — self-funding ROI, trivially justified. Enterprise seats
for the regulated segment, sold on auditability and contradiction control.
OEM/embed into agent and RAG frameworks as the memory tier. A single large
agentic customer can represent multi-million-dollar ARR while still paying a
small fraction of what GeoCell saves them.

## 13. Go-to-market: shadow mode

The wedge is the **shadow-mode replay harness** (`geocell/shadow.py`): point
GeoCell at a design partner's exported corpus + a sample of their real query
log; it runs **offline, read-only, zero LLM calls** and reports deflection
rate, agreement with their current system, the **list of stale/contradicted
answers their system is serving** (with citations), and projected $/energy
savings on their own numbers. This converts "great on our benchmarks" into
"great on *your* data, reproduced by you" — with zero production risk and a
self-evident ROI slide. Shadow → review → flip to serving is also the safe
production rollout path.

## 14. Funding milestones (see `docs/FUNDING.md`)

- **Tier 1 (pre-seed, the immediate goal):** one real customer corpus with
  customer-confirmed contradictions caught + a dollar figure on their data;
  the real-embedding result showing deflection holds on paraphrase
  (≥80% deflection at ≥99% precision); 2–3 design-partner LOIs.
- **Tier 2 (seed):** a live deployment with deflection/precision stable over
  8–12 weeks of corpus churn; head-to-head vs vector-DB RAG and a temporal
  KG on the customer's data; an instrumented cost/energy measurement.
- **Tier 3 (Series A):** multiple paying customers, integration ecosystem,
  a measured productivity claim.

## 15. Roadmap to production (see `docs/PRODUCTION.md`)

- **Phase A (~3–4 eng-weeks) "deployable in shadow":** CI + golden gates,
  numpy memory layout, WAL persistence, `forget()` with tombstones, HTTP
  API with auth/namespaces, metrics.
- **Phase B (~4–6 weeks) "deployable in serving":** incremental trust,
  background "sleep" cycles, single-writer/multi-reader, calibrated
  abstention dial + drift monitor, embedding-encoder default, extraction
  adapters, declarative authority policy.
- **Phase C (ongoing):** 1M-cell tuning, multi-tenant hardening,
  LangChain/LlamaIndex/MCP integrations, snapshot migrations.

## 16. Risks (and honest answers)

- *"Embeddings + a re-ranker already do retrieval."* True for recall; the
  wedge is the belief layer on top, proven to work on any encoder. Must be
  demonstrated on real customer data, not asserted.
- *"Is the contradiction win real or synthetic?"* The crux. The PubMed
  probe starts the answer; a customer-confirmed contradiction list finishes
  it. The shadow harness is built precisely to generate that.
- *"Why won't an LLM or vector-DB vendor add this?"* Defensibility is the
  deterministic epistemology and abstention calibration plus focus and
  speed-to-market — not the storage.
- *Prototype maturity.* Fundable as "we know exactly what to harden and
  why," with a weeks-not-months Phase A and zero-risk shadow rollout.

---

## 17. Bottom line

GeoCell is a deterministic, CPU-only memory layer that, on reproducible
benchmarks, simultaneously raises answer accuracy above the model alone,
eliminates the majority of model calls (and far more in agentic regimes by
flattening O(T²) to O(T)), and catches contradictions and hallucinations
that both LLMs and vector stores miss — while citing every answer. The
technology thesis is de-risked; the one real weakness is a pluggable
front-end; the remaining work is market proof and standard hardening, both
carried by a zero-risk shadow-mode wedge. The cost and energy implications
at market scale are material and, for long-horizon agents, potentially
decisive.

---

### Appendix — reproducibility

```bash
pip install numpy networkx pytest
python -m pytest tests/                  # 49 tests
python evals/test1_coding.py             # coding benchmark
python evals/test2_knowledge.py          # knowledge benchmark
python evals/test3_pubmed.py             # real-data probe
python evals/test4_adversarial.py        # poisoning resistance
python evals/energy_model.py             # market energy model
python evals/agentic_efficiency.py       # long-context/agentic model
python -m geocell.shadow evals/shadow_sample/corpus.jsonl \
                         evals/shadow_sample/queries.jsonl   # GTM harness
```

Supporting documents: `docs/PAPER.md` (research write-up), `docs/POSITIONING.md`,
`docs/FUNDING.md`, `docs/PRODUCTION.md`.
