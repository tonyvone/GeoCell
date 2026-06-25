# GeoCell Field v0.5

A local, CPU-only **epistemic memory engine**. No neural network, no training,
no API calls — just deterministic geometry, graph dynamics, and belief revision.

A normal database stores rows. A vector store ranks by similarity. GeoCell
treats memory as **physics**: every memory is a particle in a high-dimensional
field, and *evidence exerts force*. Corroboration attracts, contradiction
repels, trust flows, beliefs are displaced rather than deleted, and the system
writes its own inferred conclusions back into the field — with provenance —
where they can later be defeated by observation.

## Run on Replit (one click)

1. In Replit: **Create Repl → Import from GitHub** and point it at this repo.
2. Press **Run**. Replit reads `.replit` / `requirements.txt`, installs
   numpy + networkx, and starts the web app.
3. The **web preview** opens an interactive GeoCell: load the demo corpus,
   add facts with provenance, ask questions, and watch supersession,
   citations, and contradiction handling live.

No GPU, no API keys, no build step — pure Python stdlib server + numpy.

## Install & run locally

```bash
pip install -r requirements.txt
python app.py              # web demo at http://localhost:8080
python -m geocell          # or the interactive CLI / REPL
```

```bash
pip install pytest && python -m pytest tests/   # 49 tests
```

## Quick test

```text
demo
infer
settle
ask What is the current Project Orion budget?
ask Does Project Orion depend on lower compute cost?
timeline Project Orion budget
why 0
benchmark
```

## The model

Each memory is a **GeoCell**:

| facet | meaning |
|---|---|
| position | point on the unit hypersphere (deterministic hyperdimensional encoding with role binding: subject, relation triples, magnitude buckets, negation axis) |
| lexical anchor | the original encoding of the content; the cell is tethered to it forever |
| radius | semantic reach — tightens under corroboration, widens under dispute |
| trinary edges | support / unknown / contradiction to neighboring cells |
| provenance | source, date, confidence, authority |
| kind | `observed`, `inferred` (created by the engine), `concept` (consolidation) |
| status | `active`, `contested`, `superseded`, `retracted` |
| ledger | append-only audit trail of everything that ever happened to the belief |

## Six mechanics that make it formidable

**1. Belief lifecycle, not overwrites.** When a dated revision arrives
(`"budget was revised to $2.7M"`), the older claim — and stale copies of it —
are *superseded*: kept as history, down-weighted in recall, retrievable by
historical queries ("what did the old memo say?"). Conflicts with no clear
winner leave both sides *contested* and surfaced in every relevant answer.

**2. Trust propagation.** A damped fixed-point iteration: trust flows along
support edges, unresolved contradictions exert pressure from the more-trusted
side, superseded and retracted cells are capped. `why <id>` shows the exact
breakdown (prior, support flow, contradiction pressure) for any belief.

**3. Field relaxation (`settle`).** Support edges pull cells together with
force proportional to weight × mutual trust; open contradictions push apart;
a spring tethers every cell to its lexical anchor so meaning is never lost.
After settling, geometric distance encodes *corroboration*, not just wording.

**4. Spreading-activation recall.** Queries seed energy geometrically, then
the energy diffuses two hops along support edges — so a query about "compute
cost" lights up "Project Orion" even though no memory connects those words.

**5. Defeasible inference (`infer`).** Transitive rules over extracted
relations (`part_of` ∘ `depends_on` ⇒ `depends_on`, …) generate **hypothesis
cells** that live in the field like any memory: they answer questions, carry
a derivation chain, inherit discounted trust from their parents — and are
automatically *retracted* the moment they contradict observed evidence.

**6. Consolidation (`sleep`).** Communities of mutually supporting memories
merge into concept cells at the cluster centroid — abstractions used for
navigation and spreading, never as direct answers.

## Shadow-mode replay (go-to-market tool)

`python -m geocell.shadow corpus.jsonl queries.jsonl --out report.json`
runs GeoCell over a corpus + real query log, **offline, read-only, zero
LLM calls**, and reports deflection rate, agreement with the current
system, the **stale/contradicted answers it found** (with citations), and
projected $/energy savings. See `evals/shadow_sample/`.

## v0.5: hybrid retrieval (geometry + BM25) and confident abstention

The hash geometry weights every shared word equally, so a common subject
("Project Orion") drowned out the discriminating term ("vendor",
"owned"). v0.5 adds a **BM25 lexical channel** over an inverted index,
blended with the geometric cosine: rare, informative words now break
ties. A conservative **stemmer** feeds additive encoder co-features so
morphological variants (owns/owned/owner) resonate. Recall also exposes a
top-1-vs-top-2 **margin**, which callers use for *selective answering* —
serve when one candidate clearly wins, defer when it's a near-tie.

Measured on the enterprise eval (`evals/`), these three changes moved
GeoOracle from 78/100 to **100/100** answer accuracy (vs 80/100 for the
LLM alone), lifted LLM-call avoidance from 54% to **69%**, and took
citation accuracy to **100%** — every served answer is correct and cited,
ambiguous ones cleanly defer to the LLM.

## v0.4: the geometry serves (LLM replacement & enhancement layer)

**Briefings without a language model.** `brief <query>` composes a
multi-fact, citation-grounded mini-report directly from the field's
epistemic structure: the lead claim is *hedged according to its propagated
trust* (geometry decides the wording), inferred conclusions carry their
derivation, disputes are presented as disputes, displaced history is
narrated. Deterministic, milliseconds, zero tokens.

**96 bytes per memory.** `quantize` snaps every position and anchor to a
SimHash sign signature: 768 floats (3 KB) become 96 bytes — a measured
32× smaller serving index with benchmark accuracy preserved (queries are
scored asymmetrically at full precision against the quantized cells).
`savec`/`loadc` write a compact snapshot that also prunes the support
lattice to its epistemic skeleton (every contradiction and derivation
edge kept, only the strongest support links). Measured on a 400-cell
field: 17.4 MB → 488 KB on disk (36×), recall 14× faster. A snapshot
loads back *alive*: new evidence still wires, contests, and supersedes.

**GeoOracle: the cache that argues back.** Wrap any `llm(prompt) -> str`
callable in `GeoOracle(field, llm)`. Resonant queries are answered from
geometry at zero cost; novel queries escalate to the LLM and the answer
is absorbed into the field, so the hit rate climbs. Because LLM answers
enter the belief lifecycle like any memory, an answer that contradicts
higher-trust memory is **superseded on arrival** — the oracle returns the
stronger belief, flags the LLM claim as quarantined, and keeps the full
audit trail. A deterministic hallucination tripwire, no judge model.

```python
from geocell import GeoCellField
from geocell.oracle import GeoOracle

oracle = GeoOracle(field, llm=my_llm_callable)
out = oracle.query("Orion spending plan total?")
# => {"origin": "field_override", "quarantined": True,
#     "answer": "Project Orion budget was revised to $2.7 million.",
#     "llm_claim": "Project Orion budget is $9 million.", ...}
oracle.report()   # hit rate, cost avoided, quarantine count
```

## What `ask` returns

Not just an answer: the **belief status** of the answer, its propagated trust,
the evidence ranking, the superseded history it displaced, any open
contradictions touching the topic, and — if the answer was inferred — the full
reasoning chain back to observed sources.

## Commands

`demo` · `ingest <fact> | source= | date= | confidence= | authority=` ·
`recall` · `ask` · `brief` · `contradictions` · `timeline <subject>` ·
`path <a> => <b>` · `inspect <id>` · `why <id>` · `infer` · `settle` ·
`trust` · `sleep` · `quantize` · `stats` · `benchmark` ·
`save/load <path>` · `savec/loadc <path>` · `quit`

## Benchmark

`benchmark` builds a fresh field from an adversarial corpus (a revised figure,
a stale archive copy, a two-source price dispute, multi-hop-only facts) and
scores both QA accuracy and the epistemic state the field must converge to
(supersession, contested disputes, generated hypotheses, trust ordering).

## Important limitation

This is not AGI and not a production database. It is a test harness for the
GeoCell thesis: memory as geometry, contradiction as a native relation,
belief as a lifecycle, and reasoning as stable navigation across a living
memory structure.
