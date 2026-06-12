# GeoCell: Production Readiness Assessment

*What stands between the current prototype (v0.5, 49 tests, all benchmarks
green) and something a customer can run in production. Grounded in the
actual code, with acceptance criteria and rough effort.*

---

## 0. Definition of "production ready"

For the lead use case (GeoOracle as a verified memory layer in front of an
LLM, deployed via shadow mode first), production-ready means:

| Dimension | Acceptance criterion |
|---|---|
| Latency | recall/ask p95 < 100 ms at 1M cells, single CPU core |
| Ingest | sustained ≥ 200 claims/s with queries running concurrently |
| Durability | crash-safe writes (WAL); restart to serving < 30 s at 1M cells |
| Memory | < 2 GB RSS at 1M cells (quantized serving form) |
| Concurrency | many readers + one writer, no torn reads |
| Erasure | delete-by-source/id API with ledger tombstones (GDPR) |
| Tenancy | isolated namespaces, per-key auth |
| Observability | metrics endpoint: deflection, latency, quarantines, drift |
| Quality | ≥99% served precision maintained by a calibrated abstention dial |
| Release | CI gates (tests + golden benchmarks), versioned wheels, Docker |

What is *already* production-grade and should be preserved as the selling
point: determinism (same inputs → same field, bit-for-bit), the per-cell
audit ledger, confident abstention, the compact snapshot format, and the
shadow harness — which is itself the safe rollout pattern (run in shadow →
review the report → flip to serving).

---

## 1. Engine hardening (the biggest real gaps)

**1.1 Memory layout.** Cells store `position` as a Python list of 768
floats inside a dataclass — ~3.7 GB RSS at 50k cells. Move positions and
anchors into contiguous numpy arrays owned by the field (cells hold an
index), with the quantized bits as the serving default. Expected: ~10–20×
RAM reduction; required for the 1M-cell criterion.

**1.2 Incremental trust.** `propagate_trust()` is a global fixed-point
recompute, triggered by a dirty flag on the *first read after any write*.
At 50k cells / 4.4M edges that is seconds of tail latency injected into an
unlucky query. Replace with localized updates: a new cell perturbs trust
only in its graph neighborhood (bounded-radius re-iteration), with a
periodic full recompute as a background job. This is the single largest
predictable-latency risk in the current engine.

**1.3 Background dynamics.** `settle()` (Python loop over all edges ×
steps), `hypothesize()` (scans all cells), and `consolidate()` (global
community detection) must become incremental and/or scheduled maintenance
jobs ("sleep cycles"), never on the query path. Settle should operate on
the pruned epistemic skeleton with sparse matrix ops.

**1.4 Deletion and decay.** Cells only accumulate. Production needs:
`forget(source=…)` / `forget(id=…)` with ledger tombstones (erasure that
is itself audited), optional TTL/decay for low-trust dormant cells, and
index/graph cleanup. Without this there is no GDPR story and no bound on
growth.

**1.5 Concurrency.** No locks anywhere; ingest mutates the graph, postings,
and bit arrays non-atomically. Adopt single-writer / multi-reader with
copy-on-write snapshots of the serving structures (the quantized form is
naturally immutable — rebuild-and-swap). This is simpler and safer than
fine-grained locking and matches the read-heavy workload.

**1.6 Durable storage.** Today: JSON snapshot, all-or-nothing. Production:
append-only write-ahead log of ingest/supersede/retract events (the ledger
is already event-shaped — make it the source of truth), periodic compact
snapshots, replay on restart. An embedded store (SQLite/LMDB) is
sufficient; no server dependency.

**1.7 Snapshot versioning.** A `version` int exists but no migration path.
Define a schema-versioned format with forward migration and a compatibility
test in CI.

---

## 2. Intelligence quality (the proven weak front-end)

**2.1 Embedding encoder as the default option.** The seam exists and is
tested (`GeoCellField(encoder=…)`); the hash encoder stays as the
zero-dependency fallback. Ship a local sentence-embedding default and
re-run Tests 2/3 — this closes the paraphrase gap that bounds deflection.
*Blocked only by model access in the current sandbox; half a day once
unblocked.*

**2.2 Entity/claim extraction.** The `ingest(subject=…)` seam exists and is
proven to restore contradiction detection on real prose (PubMed
experiment). Ship an extraction adapter interface with two impls: a
spaCy/scispaCy NER and an optional LLM-extraction pass (extraction may use
a model; the *lifecycle* stays deterministic). Multi-language follows the
same seam.

**2.3 Calibrated abstention.** Turn the resonance/margin thresholds into a
single operator dial backed by a reliability curve measured on a held-out
slice of the customer's own replay data: "serve at ≥X% precision" →
thresholds derived, drift-monitored, re-calibrated on schedule. This is
both an ops requirement and a differentiating feature.

**2.4 Authority policy.** Today `authority` is whatever the caller passes —
the trust model's root assumption is unmanaged. Production needs a
declarative source→authority policy (per connector/domain/author role),
versioned and auditable, so "the board filing outranks the wiki" is a
reviewed config, not a code path. This is the governance feature regulated
buyers will diligence hardest.

---

## 3. Serving and operations

- **API service**: a thin HTTP layer (FastAPI) over `ingest / ask / brief /
  why / timeline / contradictions / forget`, with API-key auth and
  per-tenant namespaces (one field per namespace; the engine is already
  cheap enough to shard by tenant).
- **Observability**: structured logs; Prometheus metrics for deflection
  rate, served precision proxy (margin distribution), p50/p95 latency,
  quarantine count, open contradictions, field size; trace IDs through
  oracle → field.
- **Limits and failure modes**: request size caps, ingest rate limits,
  graceful degradation (if the field errors, the oracle falls through to
  the LLM — fail-open for availability, with the event logged).
- **Config**: thresholds, caps, authority policy, encoder choice — all in
  one versioned config object, hot-reloadable.

---

## 4. Release engineering & QA

- **CI** (none exists in the repo today): tests on push, plus *golden
  gates* — the benchmark suite (Test 1/2 QA + structural checks, shadow
  sample) must stay green for merge; scaling smoke (10k cells) nightly.
- **Property-based and fuzz tests** on the text layer (number parsing,
  subject extraction, relation extraction survived three real bugs this
  week — exactly the layer fuzzing protects) and on snapshot round-trips.
- **Packaging**: versioned wheels, a Dockerfile, semver with changelog;
  pin and test the minimal dependency set (numpy, networkx; networkx is
  replaceable by adjacency dicts already half-built — consider dropping
  it for fewer moving parts).
- **Security review**: the eval-side patch gate is not engine security;
  do a pass on deserialization (snapshot loading), resource exhaustion
  (adversarial ingest of pathological tokens), and the poisoning model
  beyond authority flooding (e.g. subject-squatting).

---

## 5. Sequenced plan

**Phase A — "deployable in shadow" (~3–4 engineer-weeks).**
CI + golden gates; numpy memory layout; WAL persistence + restart;
forget() with tombstones; HTTP API with auth/namespaces; metrics. Exit
criterion: a design partner runs shadow mode as a container against live
exports for two weeks without intervention.

**Phase B — "deployable in serving" (~4–6 weeks).**
Incremental trust; background sleep cycles; single-writer/multi-reader;
calibrated abstention dial + drift monitor; embedding encoder default;
extraction adapters; authority policy. Exit criterion: flip one partner
from shadow to serving at ≥99% measured precision and agreed deflection.

**Phase C — "scale and ecosystem" (~ongoing).**
1M-cell tuning (sparse/native inner loops), multi-tenant hardening,
LangChain/LlamaIndex/MCP integrations, snapshot migrations, SOC2-track
logging. Exit criterion: second and third tenants onboard without code
changes.

---

## 6. The honest one-liner

The *epistemics* are production-grade in spirit already — deterministic,
audited, abstaining, poisoning-resistant. What is not production-grade is
everything around them: memory layout, write-path latency, durability,
deletion, concurrency, serving surface, and the unmanaged authority input.
None of these are research risks; all are known engineering with clear
acceptance tests. Phase A is weeks, not months, and shadow mode means the
first production deployment carries near-zero risk to the customer.
