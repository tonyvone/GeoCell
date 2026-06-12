# GeoCell: A Deterministic, Geometry-Native Epistemic Memory Layer for Reducing LLM Compute

**Abstract.** Large language models are increasingly deployed over corpora
that change, where two failure modes dominate: they cannot tell a current
fact from a superseded one, and every answer — including trivially repeated
ones — costs a full, energy-intensive model inference. We present
**GeoCell**, a CPU-only, deterministic memory engine that sits in front of
an LLM. Memory is modeled as a particle field: each fact is a point on a
high-dimensional hypersphere carrying provenance, a propagated trust score,
and a belief status (active / contested / superseded / retracted).
Corroboration attracts and contradiction repels; a damped fixed-point
iteration propagates trust; defeasible transitive inference writes derived
facts back into the field; and a hybrid geometric+BM25 retriever with
confident abstention decides what can be answered from memory versus
escalated to the model. On a 100-question enterprise knowledge benchmark,
a GeoCell front layer ("GeoOracle") answers **100/100 questions correctly
versus 80/100 for the LLM alone while issuing 69% fewer model calls** and
citing a source for every served answer; it catches **10/10 contradiction
traps the LLM misses (0/10)** and quarantines fabricated answers with no
secondary judge model. On a 21-bug coding benchmark it fixes **52% of
near-repeat bugs with no model call**. We give a transparent top-down
energy model indicating that, at the measured deflection rate and moderate
adoption, a GeoCell-style layer could avoid on the order of **4–29 TWh of
data-center electricity per year (2025–2030)**. All code and benchmarks are
included and reproducible.

---

## 1. Introduction

The cost of serving LLMs is now dominated by inference, not training: at
billions of queries per day, per-query energy and dollars compound into the
largest operational expense of frontier-model providers [IEA2025, OpenAI2025].
Yet a large fraction of production queries are *grounded recall* — questions
whose answer already exists, verbatim or paraphrased, in a corpus the
operator controls — or *near-repeats* of work done before. Paying a full
model inference for these is wasteful on three axes: **money** (tokens),
**latency** (seconds), and **energy** (watt-hours).

A second, orthogonal problem is *epistemic*. Production knowledge changes:
budgets are revised, owners reassigned, contracts superseded. A similarity
retriever (vector database) will return a stale fact next to a current one
with equal confidence, and a bare LLM will confidently assert whichever it
saw most. Neither has a native notion of *which fact is true now*, *what it
displaced*, or *what is disputed*.

GeoCell addresses both with a single artifact: a deterministic memory layer
whose primitive is neither a database row nor an LLM token, but a
**deformable geometric memory object** with an epistemic lifecycle. Our
contributions:

1. A memory model that treats facts as particles in a trust-weighted field,
   with contradiction as a *native* relation and supersession instead of
   overwrite (Sections 3–4).
2. A fully deterministic, LLM-free epistemic stack — conflict detection,
   belief revision, trust propagation, defeasible inference — that is
   reproducible and auditable (Sections 4–5).
3. A serving path combining 32×-smaller binary quantization, a hybrid
   geometric+BM25 retriever, and confident abstention, packaged as an
   LLM front layer with a hallucination tripwire (Section 6).
4. Reproducible benchmarks and a transparent compute/energy impact model
   (Sections 7–9).

---

## 2. Related Work

**Truth-maintenance and belief revision.** GeoCell's trinary relations and
belief statuses descend from Truth Maintenance Systems [Doyle1979] and the
AGM theory of belief revision [AGM1985]: supersession is a restricted,
provenance-driven contraction+expansion. Unlike classical TMS, GeoCell
embeds this in a continuous geometric retrieval substrate.

**Spreading activation and vector-symbolic memory.** Recall diffuses
activation along support edges, after Collins & Loftus [Collins1975] and
ACT-R [Anderson2004]. The encoder is a hashing vector-symbolic /
hyperdimensional representation [Kanerva2009] with role binding, requiring
no trained weights.

**Lexical and geometric retrieval.** The serving retriever blends a
geometric cosine with Okapi BM25 [Robertson2009]; the binary signature path
is SimHash/LSH for cosine [Charikar2002] with asymmetric (ADC-style)
scoring [Jegou2011], i.e. the model-quantization trick applied to memory.

**Temporal knowledge graphs for agent memory.** Zep/Graphiti
[Rasmussen2025] maintain bi-temporal graphs and invalidate (not delete)
superseded edges — close in spirit, but they invoke an *LLM* to detect
contradictions. GeoCell's contradiction handling is deterministic
arithmetic, which is what makes it free, reproducible, and auditable.

**Caching and RAG.** Semantic caches return what was stored; RAG retrieves
context for the model to read. GeoCell differs by returning *what is true
now* (supersession), *flagging what is disputed*, *rejecting a wrong write*
(quarantine), and *abstaining* when uncertain.

---

## 3. The GeoCell Model

A **GeoCell** is a record ⟨content, position, radius, source, date,
confidence, authority, subject, values, polarity, kind, status, trust,
parents, ledger⟩.

**Encoding.** A sentence is mapped to a unit vector `v ∈ ℝ^d` by hashing
surface features (uni/bi/tri-grams), role-bound features (the subject and
extracted relation triples occupy their own coordinates), order-of-magnitude
buckets for numeric claims, and a negation axis; a conservative stemmer adds
morphological co-features so *owns/owned/owner* share a slot. The map is
deterministic — identical across runs and machines — so the geometry is
stable and contains no trained weights.

**Relations.** For cells *a,b* with cosine *s*, GeoCell assigns a trinary
relation: **contradiction** if they share a subject and a predicate but
conflicting numeric/polarity claims; **support** if `s ≥ τ` or they share a
subject; **unknown** otherwise. Edges carry a weight and a reason.

**Lifecycle.** A new dated claim that conflicts with an older one
*supersedes* it when it carries a revision signal or strictly higher
authority; the loser is retained as history (status `superseded`), not
deleted. Equal-strength conflicts leave both `contested`. A derived
(inferred) claim that conflicts with an observation is `retracted`. Every
transition is appended to the cell's ledger.

---

## 4. Field Dynamics

**Trust propagation.** Let `p_i = 0.55·conf_i + 0.45·auth_i` be a prior
(inferred cells inherit a discounted parent prior). Trust is the fixed point
of a damped iteration over support edges (weight `w`) and unresolved
contradictions:

```
support_i  = Σ_j w_ij · t_j  /  Σ_j w_ij              (support neighbors)
pressure_i = max_j ( t_j − t_i )                       (more-trusted opponents)
t_i ← clip( 0.55·p_i + 0.45·support_i − 0.30·pressure_i )
```

with caps on superseded/retracted cells. `why(id)` exposes the per-cell
decomposition (prior, support flow, contradiction pressure).

**Field relaxation.** Positions relax under forces: support edges attract
with magnitude `∝ w·(t_u+t_v)`, unresolved contradictions repel, and a
spring tethers each cell to its lexical anchor so meaning is never lost.
After relaxation, geometric distance encodes corroboration, not just wording.

**Hybrid retrieval.** Query activation seeds a geometric score (cosine to
anchor and settled position) blended with normalized BM25 over an inverted
index; energy then diffuses two hops along support edges. BM25 supplies the
term-informativeness the hash geometry lacks: a rare discriminating word
("vendor", "owned") outranks the common subject every candidate repeats.
This single change drove the largest accuracy gain in our ablation (§8).

**Confident abstention.** Recall exposes the top-1−top-2 resonance margin;
the serving layer answers only on a clear winner and defers near-ties to the
LLM. This is selective prediction [ElYaniv2010] and is why served accuracy
is 100% — the system never guesses.

---

## 5. Inference and Consolidation

**Defeasible inference.** Transitive rules over extracted relations (e.g.
`part_of ∘ depends_on ⇒ depends_on`) generate hypothesis cells that live in
the field like any memory: they answer questions, carry a derivation chain
to observed sources, inherit discounted trust, and are retracted on
contradiction with observation. This yields multi-hop answers absent from
any single document.

**Consolidation.** Communities of mutually supporting cells (greedy
modularity) are summarized into concept cells at the cluster centroid, used
for navigation and spreading, never as direct answers.

---

## 6. Serving Path

**Binary quantization.** Each settled position and lexical anchor is snapped
to a 768-bit SimHash signature (96 bytes), a measured **32×** reduction
versus float vectors; similarity becomes XOR+popcount, with full-precision
*queries* scored asymmetrically against quantized cells to preserve ranking.
100k memories occupy ~9.6 MB of index, served from CPU cache.

**GeoOracle.** Wraps any `llm(prompt)→str`. Resonant queries are answered
from memory (cited, with belief status); novel queries escalate and are
absorbed; LLM answers that conflict with trusted memory on the same
subject+predicate are **quarantined** (the trusted value and its citation
are returned) with a full audit trail and no secondary judge model.

---

## 7. Experimental Setup

Two reproducible benchmarks (`evals/`), run against a real model with all
calls disk-cached.

**Test 1 (coding).** A 6-module service with 56 seeded bugs (auth, routing,
type, config, database, dependency) in sibling groups. An LLM authors fixes
for a train split; only fixes accepted by a 44-test pytest suite are
distilled into GeoCell as generalized regex transformations. The 21
near-repeat test bugs are fixed GeoCell-first, escalating to the LLM on a
miss, behind an AST patch-safety gate.

**Test 2 (knowledge).** 215 synthetic enterprise documents (internal docs,
meeting notes, contracts/SOWs, customer records) with planted exact facts,
paraphrases, multi-document chains, and contradiction traps; 100 gold
questions (40 exact, 30 near-repeat, 20 synthesis, 10 contradiction)
answered LLM-only vs GeoOracle.

---

## 8. Results

**Knowledge Q&A.**

| Metric | LLM-only | GeoOracle |
|---|---|---|
| Accuracy | 80/100 | **100/100** |
| LLM calls | 100 | **31 (−69%)** |
| Citations (correct/served) | 0 | **69/69 (100%)** |
| Contradiction traps caught | 0/10 | **10/10** |
| Fabricated answers quarantined | — | **5/5 probes** |

By type, GeoOracle improved exact recall 30→40 and contradiction 0→10,
matched near-repeat (30) and synthesis (20) by deferring ambiguous cases to
the model. The cascade is strictly better than either component alone:
GeoCell serves 69 questions at 100% precision; the model handles the 31 it
defers.

**Ablation (GeoCell-served subset).** Adding the BM25 channel moved
synthesis from 4/10 to 10/10 served-correct and lifted exact recall to
40/40; adding margin-based abstention eliminated all 24 served-but-wrong
answers (served precision 100%) while holding deflection at 69%.

**Coding.** 11/21 (52%) near-repeat bugs fixed with **no** model call; fix
proposals at ~2.6 ms versus ~3 s for the model (~1,100×); 5/5 hallucinated
patches blocked; overall fix rate 90%. GeoCell-only fixes moved a multi-bug
repo from 33→44 passing tests.

---

## 9. Compute and Energy Analysis

Reproducible model in `evals/energy_model.py`. We use published anchors —
Gemini median text query 0.24 Wh and OpenAI average 0.34 Wh
[Ritchie2025, OpenAI2025], heavier RAG/agent calls modeled at ~5 Wh; data-
center electricity ~485 TWh (2025) → ~950 TWh (2030), AI share rising from
~12% to ~40% [IEA2025] — and our measured deflection (69%).

**Per query.** A GeoCell-served answer costs ~0.03–0.4 mWh of CPU
(25 W core, PUE 1.2), i.e. **~7,000–12,000× less energy** than the LLM call
it replaces. Net energy reduction across a served workload equals the
deflection fraction almost exactly (the CPU overhead is negligible): **~40%
at conservative 0.40 deflection, ~69% at measured deflection.**

**Top-down market estimate.** Layering conservative factors — inference ≈
55% of AI compute, grounded/repeat-shaped ≈ 40% of inference, then
deflection × adoption:

| Year | Addressable | 40% deflect | 69% deflect |
|---|---|---|---|
| 2025 | 12.8 TWh/yr | 0.5–5.1 TWh/yr | 0.9–8.8 TWh/yr |
| 2030 | 83.6 TWh/yr | 3.3–33.4 TWh/yr | 5.8–57.7 TWh/yr |

(Ranges span 10–100% adoption of addressable workloads.) At measured
deflection and 50% adoption this is **~4.4 TWh/yr in 2025 rising to ~28.8
TWh/yr in 2030** — 0.9%–3.0% of *all* data-center electricity, equivalent to
the annual residential electricity of millions of homes. Even the
single-product case (1M grounded queries/day) saves ~86 MWh–1.3 GWh/yr.

These are **estimates with explicit assumptions, not measurements**: CPU
energy is attributed, LLM per-query energy and volumes are vendor/IEA
figures, and deflection comes from a synthetic benchmark. The model is
parameterized so any input can be challenged and re-run.

---

## 9b. Scale, real data, and adversarial robustness

Following the original synthetic benchmarks we ran four further probes
(`evals/`), addressing scale, real corpora, and security.

**Scale (`evals/`, synthetic, pathologically dense — 88–220 support
edges/cell).** Replacing O(n²) ingest (all-pairs comparison) with
inverted-index candidate wiring, and dense-matrix recall with
candidate-bounded local spreading + vectorized scoring, makes both
operations bounded:

| Cells | Ingest | Recall p50 | Recall p95 | RAM |
|---|---|---|---|---|
| 10,000 | ~5 ms/cell | 153 ms | 210 ms | 0.8 GB |
| 50,000 | ~7 ms/cell | 262 ms | 302 ms | 3.7 GB |

Recall no longer grows with field size (it is bounded by the candidate
cap); the residual cost is from the synthetic data's extreme edge density,
far above natural-language corpora. This is an unoptimized pure-Python
CPU prototype with headroom (sparse linear algebra, native inner loops).

**Real-data domain probe (`evals/test3_pubmed.py`).** Four *real* C9orf72/
ALS abstracts (2011–2025; *Neuron*, *Eur. J. Neurol.*, *Lancet Neurol.*;
captured via the PubMed MCP, DOIs retained) in which the reported C9orf72
share of familial ALS genuinely drifts (23.5% → ~46% → 30–50%). Findings,
reported honestly:
- **Works on real prose, domain-agnostic, zero LLM:** per-answer citations
  to real papers; trust correctly ordered by journal authority + recency;
  fully deterministic.
- **Degrades on real prose with the built-in rule extractors:** the
  rule-based *subject* extractor returns "repeat expansion" rather than
  "C9orf72", so the numeric contradiction is missed; hash lexical recall
  misses semantic paraphrase ("protein pathology" → "TDP-43 aggregates").
- **Controlled isolation:** supplying the canonical entity through the new
  `ingest(subject=…)` seam (standing in for a domain NER) makes the
  contradiction fire and orders the sources correctly. The bottleneck is
  therefore the NLP front-end, not the epistemic layer — and the front-end
  (encoder, entity/claim extractor) is pluggable by design.

**Adversarial poisoning (`evals/test4_adversarial.py`).** Against one
high-authority truth we inject up to 500 low-authority contradictions
(a Sybil/write-poisoning flood). The truth remains the active belief and
the highest-trust cell at every flood size; trust does not collapse; the
injected claims are superseded, not adopted. Because trust is propagated
by *source authority*, not vote count, a flood of weak sources cannot
outweigh one trusted source — a structural defense, with zero LLM cost.

**Pluggability (engine).** `GeoCellField(encoder=…)` accepts any
`(text, dims) → vector` callable and `ingest(subject=…)` accepts an
external entity; both leave the belief layer (lifecycle, trust,
contradiction) untouched, since it operates on the resulting geometry and
subject keys rather than on how they were produced. (The real-embedding
ablation the roadmap calls for is implemented as this seam but was not run
here: the execution environment blocks the model download.)

## 10. Limitations

(1) The hash encoder is weak on deep semantic paraphrase ("automobile" vs
"car"); GeoCell handles this today by *deferring* such queries to the LLM
(hence 100% served accuracy but bounded deflection). The position encoder is
pluggable: real embeddings would raise deflection without touching the
belief layer. (2) Benchmarks are synthetic — realistic but not a customer
corpus. (3) Latency figures are from an unoptimized Python prototype;
ingestion is O(n²) and field relaxation is a Python loop. (4) Relation
extraction is rule-based and English-only. (5) The energy figures are a
transparent estimate, not an audited measurement.

---

## 11. Conclusion

GeoCell shows that a deterministic, CPU-only memory layer with a real
epistemology — contradiction as a native relation, supersession instead of
overwrite, trust propagation, and confident abstention — can sit in front of
an LLM and simultaneously (i) raise answer accuracy above the model alone,
(ii) eliminate the majority of model calls on grounded workloads, and (iii)
catch hallucinations and contradictions the model and a plain vector store
both miss, all while citing its sources. The compute and energy implications
at market scale are material. The most valuable next step is to replace the
synthetic corpus and hash encoder with a customer corpus and pluggable
embeddings, and to publish a measured (not modeled) energy comparison.

---

## References

- [AGM1985] Alchourrón, Gärdenfors, Makinson. *On the Logic of Theory
  Change.* J. Symbolic Logic, 1985.
- [Anderson2004] Anderson et al. *An Integrated Theory of the Mind (ACT-R).*
  Psychological Review, 2004.
- [Charikar2002] Charikar. *Similarity Estimation Techniques from Rounding
  Algorithms (SimHash).* STOC, 2002.
- [Collins1975] Collins & Loftus. *A Spreading-Activation Theory of Semantic
  Processing.* Psychological Review, 1975.
- [Doyle1979] Doyle. *A Truth Maintenance System.* Artificial Intelligence,
  1979.
- [ElYaniv2010] El-Yaniv & Wiener. *On the Foundations of Noise-free
  Selective Classification.* JMLR, 2010.
- [IEA2025] International Energy Agency. *Energy and AI / Key Questions on
  Energy and AI.* 2025. https://www.iea.org/reports/energy-and-ai
- [Jegou2011] Jégou, Douze, Schmid. *Product Quantization for Nearest
  Neighbor Search.* IEEE TPAMI, 2011.
- [Kanerva2009] Kanerva. *Hyperdimensional Computing.* Cognitive
  Computation, 2009.
- [OpenAI2025] OpenAI statements on per-query energy (~0.34 Wh), 2025; see
  Ritchie2025 for synthesis.
- [Rasmussen2025] Rasmussen et al. *Zep: A Temporal Knowledge Graph
  Architecture for Agent Memory.* 2025. https://arxiv.org/abs/2501.13956
- [Ritchie2025] Ritchie. *The carbon/electricity footprint of ChatGPT and
  Gemini (2025 update).* https://hannahritchie.substack.com/p/ai-footprint-august-2025
- [Robertson2009] Robertson & Zaragoza. *The Probabilistic Relevance
  Framework: BM25 and Beyond.* Foundations and Trends in IR, 2009.

*Artifacts: `geocell/` (engine, 42 tests), `evals/` (benchmarks,
energy model). All numbers reproducible via `pytest tests/`,
`python evals/test1_coding.py`, `python evals/test2_knowledge.py`,
`python evals/energy_model.py`.*
