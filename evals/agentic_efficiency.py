"""Efficiency model for long-context prompts and agentic flows.

The flat per-query deflection number understates GeoCell badly in two
regimes where token cost compounds super-linearly:

  Mechanism 1 -- RAG context compression. Naive RAG stuffs k retrieved
    chunks into every prompt. GeoCell either answers outright (0 tokens)
    or escalates with ONE verified, cited fact instead of k chunks -- so
    even the escalated fraction reads far fewer input tokens.

  Mechanism 2 -- Agentic context growth. In a T-step agent loop, the
    naive pattern re-sends the whole growing transcript every step, so
    cumulative input tokens are O(T^2). If GeoCell holds the memory and
    the agent keeps a bounded working window (querying memory on demand),
    per-step context is ~constant and cumulative cost is O(T).

  Mechanism 3 -- Deflection. The fraction of sub-queries / tool lookups
    answered from memory with no model call at all (measured: 0.69).

This is a transparent, parameterized ESTIMATE. Token counts and prices are
the defensible primary outputs; energy is a scaled-from-tokens secondary
estimate with a stated per-token figure. We compare against a baseline
that USES prompt caching (static prefix cached), to be conservative --
GeoCell's win is on the dynamic, uncacheable, growing part of the context.

Run:  python evals/agentic_efficiency.py
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- pricing & energy anchors (override freely) --------------------------
INPUT_USD_PER_M = 3.0       # $/1M input tokens (mid-tier frontier model)
OUTPUT_USD_PER_M = 15.0     # $/1M output tokens
CACHED_INPUT_USD_PER_M = 0.30   # cached-prefix read (~10x cheaper)
WH_PER_1K_TOKENS = 0.30     # energy per 1k processed tokens (≈0.34 Wh avg query)
GEOCELL_WH_PER_SERVE = 0.0004   # ~50 ms CPU, attributed
GEOCELL_USD_PER_SERVE = 1e-6    # CPU-cents, effectively free

DEFLECTION = 0.69           # measured (evals/test2_results.json)


def cost_usd(in_tok, out_tok, cached_tok=0):
    return ((in_tok - cached_tok) / 1e6 * INPUT_USD_PER_M
            + cached_tok / 1e6 * CACHED_INPUT_USD_PER_M
            + out_tok / 1e6 * OUTPUT_USD_PER_M)


def energy_wh(in_tok, out_tok):
    # Output tokens cost ~1 forward pass each; weight them ~3x input.
    return (in_tok + 3 * out_tok) / 1000.0 * WH_PER_1K_TOKENS


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def pct(a, b):
    return f"{(1 - a / b) * 100:.1f}%" if b else "n/a"


# --------------------------------------------------------------------------
# Mechanism 1: RAG context compression (single-shot grounded query)
# --------------------------------------------------------------------------

@dataclass
class RAGParams:
    base_prompt: int = 800       # system + question
    k_chunks: int = 20           # retrieved chunks stuffed per query
    chunk_tokens: int = 400
    output_tokens: int = 150
    verified_fact_tokens: int = 200   # GeoCell's single cited answer


def rag_single(p: RAGParams):
    # Baseline: stuff k chunks. Static system prompt cacheable; the k
    # retrieved chunks are query-specific and NOT cacheable.
    base_in = p.base_prompt + p.k_chunks * p.chunk_tokens
    base = {
        "in": base_in, "out": p.output_tokens,
        "usd": cost_usd(base_in, p.output_tokens, cached_tok=p.base_prompt),
        "wh": energy_wh(base_in, p.output_tokens),
    }
    # GeoCell: deflect a fraction to 0-LLM; escalate the rest with ONE fact.
    esc_in = p.base_prompt + p.verified_fact_tokens
    esc_cost = cost_usd(esc_in, p.output_tokens, cached_tok=p.base_prompt)
    esc_wh = energy_wh(esc_in, p.output_tokens)
    geo = {
        "usd": (1 - DEFLECTION) * esc_cost + DEFLECTION * GEOCELL_USD_PER_SERVE,
        "wh": (1 - DEFLECTION) * esc_wh + DEFLECTION * GEOCELL_WH_PER_SERVE,
        "in_avg": (1 - DEFLECTION) * esc_in,
    }
    return base, geo


# --------------------------------------------------------------------------
# Mechanism 2: agentic loop -- context growth O(T^2) vs O(T)
# --------------------------------------------------------------------------

@dataclass
class AgentParams:
    steps: int = 30
    system_tools: int = 1500     # system prompt + tool definitions (cacheable)
    per_step_new: int = 700      # tool result + reasoning added each step
    output_per_step: int = 220
    working_window: int = 3      # GeoCell agent keeps last-N steps live
    retrieved_per_step: int = 200    # one memory fact pulled on demand


def agent_run(p: AgentParams):
    """Conservative: models CONTEXT COMPRESSION ONLY (every step still hits
    the LLM). The win comes purely from a bounded working window + on-demand
    memory retrieval instead of a transcript that grows every step.
    Deflection of lookup-shaped steps is an ADDITIONAL effect, reported
    separately, not folded in here."""
    base_usd = base_wh = 0.0
    base_in_total = 0
    geo_usd = geo_wh = 0.0
    geo_in_total = 0
    for t in range(1, p.steps + 1):
        # Baseline (with prefix caching): system+tools cached, but the full
        # growing dynamic transcript is uncacheable and re-read every step.
        base_in = p.system_tools + t * p.per_step_new
        base_in_total += base_in
        base_usd += cost_usd(base_in, p.output_per_step, cached_tok=p.system_tools)
        base_wh += energy_wh(base_in, p.output_per_step)
        # GeoCell: bounded window + one retrieved fact, ~constant per step.
        geo_in = p.system_tools + min(t, p.working_window) * p.per_step_new + p.retrieved_per_step
        geo_in_total += geo_in
        geo_usd += cost_usd(geo_in, p.output_per_step, cached_tok=p.system_tools)
        geo_wh += energy_wh(geo_in, p.output_per_step)
    return (base_in_total, base_usd, base_wh), (geo_in_total, geo_usd, geo_wh)


def main():
    section("Mechanism 1 — RAG context compression (per grounded query)")
    p = RAGParams()
    base, geo = rag_single(p)
    print(f"  Baseline (stuff {p.k_chunks} chunks): {base['in']:,} in + "
          f"{base['out']} out tok  ->  ${base['usd']:.5f}, {base['wh']:.4f} Wh")
    print(f"  GeoCell ({DEFLECTION:.0%} deflect, else 1 cited fact): "
          f"~{geo['in_avg']:,.0f} avg in tok  ->  ${geo['usd']:.5f}, {geo['wh']:.4f} Wh")
    print(f"  REDUCTION: cost {pct(geo['usd'], base['usd'])}, "
          f"energy {pct(geo['wh'], base['wh'])} per query")

    section("Mechanism 2 — Agentic loop (cumulative over a run)")
    for steps in (10, 30, 100):
        ap = AgentParams(steps=steps)
        (b_in, b_usd, b_wh), (g_in, g_usd, g_wh) = agent_run(ap)
        print(f"  {steps:>3}-step agent: baseline {b_in:,} in tok / ${b_usd:.4f} / "
              f"{b_wh:.2f} Wh")
        print(f"               GeoCell  {g_in:,} in tok / ${g_usd:.4f} / "
              f"{g_wh:.2f} Wh")
        print(f"               REDUCTION: tokens {pct(g_in, b_in)}, "
              f"cost {pct(g_usd, b_usd)}, energy {pct(g_wh, b_wh)}")

    section("Why it compounds: baseline input tokens grow O(T^2)")
    ap = AgentParams()
    for steps in (5, 10, 20, 40, 80):
        b = sum(ap.system_tools + t * ap.per_step_new for t in range(1, steps + 1))
        g = sum(ap.system_tools + min(t, ap.working_window) * ap.per_step_new
                + ap.retrieved_per_step for t in range(1, steps + 1))
        print(f"  T={steps:>3}: baseline {b:>9,} in tok   GeoCell {g:>8,}   "
              f"ratio {b/max(1,g):.1f}x")

    section("Additional layer: deflection of lookup-shaped steps")
    # On top of compression, the fraction of agent steps that are pure
    # factual lookups (not reasoning) can be served from memory with no LLM
    # call at all. We use a CONSERVATIVE agent-lookup rate, distinct from
    # the 0.69 measured on grounded Q&A.
    for lookup_frac in (0.2, 0.4):
        ap = AgentParams(steps=30)
        (_, b_usd, b_wh), (_, g_usd, g_wh) = agent_run(ap)
        # deflected steps cost ~nothing instead of a bounded LLM call
        g_usd2 = g_usd * (1 - lookup_frac)
        g_wh2 = g_wh * (1 - lookup_frac)
        print(f"  30-step, {lookup_frac:.0%} of steps are deflectable lookups: "
              f"cost {pct(g_usd2, b_usd)}, energy {pct(g_wh2, b_wh)} vs baseline")

    section("Scale: a fleet of agents")
    runs_per_day = 100_000
    ap = AgentParams(steps=30)
    (_, b_usd, b_wh), (_, g_usd, g_wh) = agent_run(ap)   # compression only
    d_usd = (b_usd - g_usd) * runs_per_day * 365
    d_wh = (b_wh - g_wh) * runs_per_day * 365
    homes = d_wh / 1000.0 / 3000.0      # Wh -> kWh -> homes @ 3000 kWh/yr
    print(f"  {runs_per_day:,} 30-step agent runs/day (compression only):")
    print(f"     cost saved : ${d_usd/1e6:.1f}M/year")
    print(f"     energy saved: {d_wh/1e9:.2f} GWh/year (~{homes:,.0f} US homes)")

    print("\nNOTE: estimate with explicit assumptions (pricing, chunking, step")
    print("counts). Token math is the defensible core; energy is scaled from")
    print("tokens. Baseline already uses prefix caching -- the win is on the")
    print("growing, uncacheable dynamic context. Deflection (0.69) is measured.")


if __name__ == "__main__":
    main()
