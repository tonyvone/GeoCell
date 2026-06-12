# GeoCell Field v0.3

A local, CPU-only **epistemic memory engine**. No neural network, no training,
no API calls — just deterministic geometry, graph dynamics, and belief revision.

A normal database stores rows. A vector store ranks by similarity. GeoCell
treats memory as **physics**: every memory is a particle in a high-dimensional
field, and *evidence exerts force*. Corroboration attracts, contradiction
repels, trust flows, beliefs are displaced rather than deleted, and the system
writes its own inferred conclusions back into the field — with provenance —
where they can later be defeated by observation.

## Install & run

```bash
pip install numpy networkx
python -m geocell          # or: python geocell_lab.py
```

```bash
pip install pytest && python -m pytest tests/   # 22 tests
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

## What `ask` returns

Not just an answer: the **belief status** of the answer, its propagated trust,
the evidence ranking, the superseded history it displaced, any open
contradictions touching the topic, and — if the answer was inferred — the full
reasoning chain back to observed sources.

## Commands

`demo` · `ingest <fact> | source= | date= | confidence= | authority=` ·
`recall` · `ask` · `contradictions` · `timeline <subject>` ·
`path <a> => <b>` · `inspect <id>` · `why <id>` · `infer` · `settle` ·
`trust` · `sleep` · `stats` · `benchmark` · `save/load <path>` · `quit`

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
