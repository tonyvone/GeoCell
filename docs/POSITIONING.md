# GeoCell: A Verified Memory Layer for AI Systems

*Positioning, practical use, and measured efficiency — written against the
numbers in `evals/`, not aspiration.*

---

## 1. The one-paragraph version

GeoCell is a CPU-only, deterministic memory engine that sits **in front of
an LLM**. It answers from memory whatever it can answer with high
confidence — instantly, for free, with a citation — and escalates only the
genuinely novel questions to the model. Because memory is an *epistemic*
structure (every fact has provenance, trust, and a belief status), GeoCell
does two things a vector database cannot: it **catches the LLM when it
contradicts what the company already knows**, and it **shows its work**. In
our enterprise knowledge benchmark it answered **100/100 questions
correctly versus 80/100 for the LLM alone, while making 69% fewer model
calls** and citing a source for every answer it served.

---

## 2. How it's used in practice

GeoCell is not an agent and not a chatbot. It's a **layer**. Three
deployment shapes, all of which we have working code for:

### A. The verified-answer cache (GeoOracle)

Wrap any LLM call. GeoCell intercepts the query:

```python
oracle = GeoOracle(field, llm=your_model_call)
out = oracle.query("What is the current Project Orion budget?")
# -> served from memory, $0, 50ms, cited, belief_status="active"
```

- **Resonant query** → answered from memory: zero tokens, sub-100ms, with
  a source citation and a belief status (active / contested / superseded).
- **Novel query** → escalated to the LLM, and the answer is **absorbed**
  back into memory, so the hit rate climbs with use.
- **LLM answer that contradicts trusted memory** → quarantined. The oracle
  returns the trusted value, flags the model's claim, and keeps an audit
  trail. No second "judge" model required.

**Who needs this:** any product doing RAG or agentic Q&A over a corpus that
changes — internal knowledge assistants, customer-support copilots,
financial/operations analysts, compliance tooling.

### B. The repeat-work memory for coding agents

A coding agent burns tokens re-deriving the same class of fix. GeoCell
learns a verified fix once (only fixes that pass the test suite are
stored), generalizes it, and applies it to near-repeat bugs **with no model
call** — gated by a static safety check that refuses hallucinated code
(bad imports, non-existent stdlib calls, syntax errors).

**Who needs this:** autonomous coding agents, CI auto-fix bots, large
monorepo refactors where the same mistake recurs across modules.

### C. The system-of-record memory for regulated workflows

The belief lifecycle — supersession instead of overwrite, contested
disputes surfaced, a per-fact ledger, trust propagation — is itself the
product. An assistant that can say *"this is contested, here are both
sources, here's why the newer one wins"* is the difference between a demo
and something legal/finance/healthcare can deploy.

**Who needs this:** anywhere a wrong-but-confident answer is a liability,
not an annoyance.

---

## 3. The efficiency gain vs LLM-only

All figures below are from the two reproducible benchmarks in `evals/`,
run against a real model with cached calls.

### Knowledge Q&A (100 questions over a 215-document corpus)

| Metric | LLM-only | GeoCell + LLM (GeoOracle) | Delta |
|---|---|---|---|
| Correct answers | 80 / 100 | **100 / 100** | **+25% accuracy** |
| LLM calls made | 100 | **31** | **−69% model calls** |
| Answers with a citation | 0 | 69 (100% correct) | auditability |
| Contradiction traps caught | 0 / 10 | **10 / 10** | the headline |
| Answer latency (served) | ~3 s | **~50 ms** (≈4 ms quantized) | **60–750× faster** |

**What "−69% model calls" means for cost.** Inference is billed per token.
GeoCell removes 69% of the calls on this workload outright. The served
answers cost **fractions of a cent of CPU** instead of a model round-trip.
At 1M queries/month, a 69% deflection turns (illustratively) a $10k/month
inference bill into ~$3.1k — and the deflected 69% also return in
milliseconds instead of seconds, which is its own conversion/UX win.

### Coding fixes (21 near-repeat bugs)

| Metric | LLM-only | GeoCell-first | 
|---|---|---|
| Bugs fixed with **no** model call | 0 | **11 / 21 (52%)** |
| Fix latency | ~3 s (model) | **2.6 ms** (≈1,100× faster) |
| Hallucinated patches blocked | n/a | **5 / 5** |
| Overall fix rate | 82% | 90% |

GeoCell-only fixes moved a multi-bug repo from 33→44 passing tests with
zero model calls.

### Storage efficiency

Memories serve from a **96-byte binary signature** each — a measured **32×**
smaller than float vectors. 100,000 memories fit in **9.6 MB** of index
versus 307 MB. This is the model-quantization trick applied to memory: the
serving index runs in cache, on a CPU, with no GPU anywhere in the stack.

### The honest bound

GeoCell does not replace open-ended generation. It deflects the **repeat
and grounded-recall** fraction of a workload — which in practice is large —
and makes the LLM the fallback, not the front line. The 31% it escalates
still cost full price. The win is removing the other 69% and catching the
model when it's wrong.

---

## 4. Why it wins where it wins (the technical moat)

Three things competitors structurally don't do:

1. **Contradiction as a native operation.** Vector stores rank by
   similarity and will happily return a stale fact next to a current one.
   Temporal knowledge graphs (Graphiti/Zep) handle this but call an LLM to
   detect the conflict. GeoCell's contradiction detection, supersession,
   and trust propagation are **deterministic arithmetic** — reproducible,
   auditable, free. The baseline LLM got **0 of 10** contradiction traps;
   GeoCell got **10 of 10**.

2. **A hallucination tripwire with no judge model.** Because an LLM answer
   enters the same belief lifecycle as any memory, one that contradicts a
   trusted fact is rejected on arrival — and we recover the trusted value
   with its citation. 5/5 fabricated answers caught in each benchmark.

3. **Confident abstention.** GeoCell knows when it doesn't know: it serves
   only when one candidate clearly beats the runner-up, and defers
   near-ties to the LLM. That's why served accuracy is **100%** — it never
   guesses. This is the property that makes a cache trustworthy.

---

## 5. Positioning & marketing

### Category

Don't sell it as "another vector DB." Create the category:

> **The verified memory layer for AI** — the trust and cost-control tier
> between your agents and your LLM.

### Tagline options

- *"The cache that argues back."*
- *"Memory that knows what's true — and what's changed."*
- *"Cut LLM spend. Catch hallucinations. Cite every answer."*

### The three-headline pitch

1. **Spend less.** Deflect the majority of LLM calls to a CPU memory that
   answers in milliseconds. (−69% calls, measured.)
2. **Hallucinate less.** Quarantine model answers that contradict your
   trusted knowledge — no judge model, full audit trail.
3. **Prove it.** Every served answer carries a source, a belief status, and
   a ledger. Built for the workflows where "the AI said so" isn't enough.

### Ideal customers (in priority order)

1. **Regulated knowledge work** — finance, legal, healthcare, insurance.
   They cannot ship confident-but-wrong, and they need citations and audit
   trails by mandate. The contradiction/quarantine story is the wedge.
2. **High-volume AI products** — support copilots, internal assistants —
   where a 50–70% call deflection is a direct, large line-item saving.
3. **Coding-agent platforms** — repeat-fix memory + the hallucinated-code
   gate.

### Competitive framing

| | Raw LLM | Vector DB / RAG | Temporal KG (Zep/Graphiti) | **GeoCell** |
|---|---|---|---|---|
| Grounded recall | ✗ | ✓ | ✓ | ✓ |
| Handles contradictions | ✗ | ✗ | ✓ (LLM-based) | ✓ (deterministic) |
| Catches LLM hallucination | ✗ | ✗ | partial | ✓ |
| Cuts LLM calls | ✗ | partial | partial | ✓ (−69%) |
| CPU-only, no GPU | n/a | mostly | mostly | ✓ |
| Audit trail / citations | ✗ | source only | ✓ | ✓ (+ ledger, trust) |

### Likely objections — and the honest answers

- *"Embeddings would retrieve better."* Often true on pure paraphrase — and
  GeoCell's position encoder is **pluggable**. The differentiator is the
  belief layer on top, which works with any embedding. We even defer hard
  paraphrases to the LLM today and still hit 100% served accuracy.
- *"Isn't this just a cache?"* A cache returns what it stored. GeoCell
  returns what's *true now*, supersedes what changed, flags what's
  disputed, and rejects a wrong write. It's a cache with an epistemology.
- *"How do I trust the deflected answers?"* That's the point of confident
  abstention: it only serves clear winners and cites them. The measured
  served-answer accuracy is 100%; everything uncertain goes to your model.

### Business model angles

- **Usage-based on calls deflected** — you charge a fraction of the LLM
  spend you eliminate. ROI is self-evident and self-funding.
- **Seat/enterprise** for the regulated segment, sold on auditability and
  hallucination control rather than cost.
- **OEM/embed** in agent platforms and RAG frameworks as the memory tier.

---

## 6. What to say, and what not to over-claim

**Say:** deterministic, CPU-only, cited, auditable; −69% model calls and
100% served accuracy on our benchmark; catches contradictions an LLM and a
plain vector store both miss; 32× smaller serving index.

**Don't over-claim:** it is not AGI and not a generation model; the
benchmarks are synthetic (realistic, but not a customer corpus); latency
figures are from an unoptimized Python prototype; deeply semantic paraphrase
still benefits from real embeddings (which it supports). The right framing is
a **layer that makes your existing LLM cheaper, safer, and auditable** — not
a replacement for it.
