"""Market-scale compute & energy impact model for GeoCell deflection.

This is a transparent, parameterized estimate -- NOT a measurement. Every
input is a named constant with a cited source or a clearly-labeled
assumption, so a skeptical reader can change one number and see the
result move. We deliberately use conservative (low) deflection and a
range of energy-per-query anchors.

Run:  python evals/energy_model.py

Key published anchors (mid-2025/2026, see docs/PAPER.md references):
  - Google Gemini median text query:        0.24 Wh   (Google, Aug 2025)
  - OpenAI "average" ChatGPT query:          0.34 Wh   (OpenAI, 2025)
  - ChatGPT query volume:                    ~2.5 B queries/day (Jul 2025)
  - Global data-center electricity (2025):   ~485 TWh  (IEA)
  - Projected data-center electricity (2030):~950 TWh  (IEA)

GeoCell measured inputs (this repo, evals/):
  - LLM-call deflection on grounded Q&A:     69%  (we model 40-69%)
  - CPU energy per served GeoCell answer:    measured/estimated below
"""
from __future__ import annotations

from dataclasses import dataclass


# ---- Published anchors ---------------------------------------------------
WH_PER_QUERY_LOW = 0.24      # Gemini median text query
WH_PER_QUERY_HIGH = 0.34     # OpenAI average query
# Reasoning / long-context / agentic queries are far heavier. Independent
# analyses put complex prompts 10-50x the median. We model a "mixed
# enterprise" workload as a blend.
WH_PER_QUERY_REASONING = 5.0  # conservative figure for a heavier RAG/agent call

CHATGPT_QUERIES_PER_DAY = 2.5e9   # Jul 2025

# ---- GeoCell measured / estimated inputs --------------------------------
DEFLECTION_MEASURED = 0.69    # evals/test2_results.json
DEFLECTION_CONSERVATIVE = 0.40

# Energy for one served GeoCell answer on CPU.
# Measured serving latency (quantized recall, evals): ~4 ms; full path ~50 ms.
# A server CPU core under load ~ 15-35 W; attribute one core for the answer.
CPU_W = 25.0
GEOCELL_SECONDS_QUANTIZED = 0.004
GEOCELL_SECONDS_FULL = 0.050
PUE = 1.2                      # data-center power usage effectiveness (modern)


def geocell_wh(seconds: float) -> float:
    return CPU_W * seconds / 3600.0 * PUE


@dataclass
class Scenario:
    name: str
    wh_per_llm_query: float
    deflection: float
    geocell_seconds: float

    def wh_saved_per_deflected(self) -> float:
        # We pay a tiny CPU cost to avoid one LLM query.
        return self.wh_per_llm_query - geocell_wh(self.geocell_seconds)

    def net_reduction_fraction(self) -> float:
        """Energy reduction across the *whole* served workload, given that
        deflected queries still cost a little CPU and the rest cost full."""
        served_geocell = self.deflection * geocell_wh(self.geocell_seconds)
        served_llm = (1 - self.deflection) * self.wh_per_llm_query
        total_with = served_geocell + served_llm
        total_without = self.wh_per_llm_query
        return 1 - total_with / total_without


def fmt_energy(wh: float) -> str:
    if wh >= 1e9:
        return f"{wh/1e9:.2f} GWh"
    if wh >= 1e6:
        return f"{wh/1e6:.2f} MWh"
    if wh >= 1e3:
        return f"{wh/1e3:.2f} kWh"
    return f"{wh:.3f} Wh"


def section(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def main():
    section("1. Per-query energy: LLM vs GeoCell-served")
    for s in [GEOCELL_SECONDS_QUANTIZED, GEOCELL_SECONDS_FULL]:
        print(f"  GeoCell served answer @ {s*1000:.0f} ms CPU "
              f"({CPU_W} W, PUE {PUE}): {geocell_wh(s)*1000:.4f} mWh "
              f"= {geocell_wh(s):.6f} Wh")
    print(f"  LLM query (median text):   {WH_PER_QUERY_LOW} Wh")
    print(f"  LLM query (avg):           {WH_PER_QUERY_HIGH} Wh")
    print(f"  LLM query (heavy RAG/agent): {WH_PER_QUERY_REASONING} Wh")
    ratio_low = WH_PER_QUERY_LOW / geocell_wh(GEOCELL_SECONDS_QUANTIZED)
    ratio_high = WH_PER_QUERY_REASONING / geocell_wh(GEOCELL_SECONDS_FULL)
    print(f"  => a deflected query uses ~{ratio_low:,.0f}x to "
          f"~{ratio_high:,.0f}x less energy than the LLM call it replaces")

    section("2. Net energy reduction on a served workload")
    scenarios = [
        Scenario("Conservative (0.24 Wh, 40% deflect, full path)",
                 WH_PER_QUERY_LOW, DEFLECTION_CONSERVATIVE, GEOCELL_SECONDS_FULL),
        Scenario("Measured (0.34 Wh, 69% deflect, full path)",
                 WH_PER_QUERY_HIGH, DEFLECTION_MEASURED, GEOCELL_SECONDS_FULL),
        Scenario("Heavy workload (5.0 Wh, 69% deflect, quantized)",
                 WH_PER_QUERY_REASONING, DEFLECTION_MEASURED, GEOCELL_SECONDS_QUANTIZED),
    ]
    for s in scenarios:
        print(f"  {s.name}")
        print(f"      net energy reduction on workload: "
              f"{s.net_reduction_fraction()*100:.1f}%")

    section("3. Single high-volume product (1M grounded queries/day)")
    q_per_day = 1e6
    for s in scenarios:
        daily_without = q_per_day * s.wh_per_llm_query
        daily_with = (q_per_day * s.deflection * geocell_wh(s.geocell_seconds)
                      + q_per_day * (1 - s.deflection) * s.wh_per_llm_query)
        saved_yr = (daily_without - daily_with) * 365
        print(f"  {s.name}")
        print(f"      saved/year: {fmt_energy(saved_yr)}  "
              f"(~{saved_yr/1e3/3000:,.0f} US homes-equiv at 3 MWh/home/yr)")

    section("4. Market-scale illustration (ChatGPT-volume, grounded fraction)")
    # Not all queries are deflectable. Assume a 'grounded/repeat' fraction
    # of total traffic that GeoCell-style memory can serve.
    grounded_fraction = 0.50   # half of queries are recall/repeat-shaped
    deflect = DEFLECTION_CONSERVATIVE   # be conservative at market scale
    for wh in [WH_PER_QUERY_LOW, WH_PER_QUERY_HIGH]:
        addressable = CHATGPT_QUERIES_PER_DAY * grounded_fraction
        deflected = addressable * deflect
        wh_saved_day = deflected * (wh - geocell_wh(GEOCELL_SECONDS_FULL))
        twh_yr = wh_saved_day * 365 / 1e12
        print(f"  @ {wh} Wh/query, {grounded_fraction:.0%} grounded, "
              f"{deflect:.0%} deflection:")
        print(f"      deflected: {deflected/1e9:.2f} B queries/day")
        print(f"      energy saved: {twh_yr:.2f} TWh/year "
              f"(~{twh_yr/485*100:.2f}% of 2025 data-center electricity)")

    section("4b. TOP-DOWN MARKET ENERGY IMPACT (the headline number)")
    # Layered, conservative top-down. Each factor is a labeled assumption.
    DC = {"2025": 485.0, "2030": 950.0}                  # TWh, IEA
    AI_SHARE = {"2025": 0.12, "2030": 0.40}              # AI % of DC power
    INFERENCE_SHARE = 0.55                                # inference % of AI compute
    GROUNDED = 0.40                                       # recall/repeat-shaped % of inference
    for year in ("2025", "2030"):
        ai_twh = DC[year] * AI_SHARE[year]
        infer_twh = ai_twh * INFERENCE_SHARE
        addressable_twh = infer_twh * GROUNDED
        print(f"  {year}: data-center {DC[year]:.0f} TWh "
              f"-> AI {ai_twh:.0f} TWh -> inference {infer_twh:.0f} TWh "
              f"-> GeoCell-addressable {addressable_twh:.1f} TWh")
        for deflect, dname in [(DEFLECTION_CONSERVATIVE, "40% deflect"),
                               (DEFLECTION_MEASURED, "69% deflect")]:
            for adoption in (0.10, 0.50, 1.00):
                saved = addressable_twh * deflect * adoption
                print(f"        {dname}, {adoption:.0%} adoption: "
                      f"save {saved:.1f} TWh/yr "
                      f"({saved/DC[year]*100:.2f}% of all DC power; "
                      f"~{saved*1e9/3000/1e6:.1f}M US homes)")

    section("5. Cost translation (illustrative)")
    # API list prices vary widely; we anchor on a representative blended
    # $/query rather than tokens to keep it legible.
    for price in [0.001, 0.005, 0.02]:
        q = 1e6
        saved_day = q * DEFLECTION_MEASURED * price
        print(f"  @ ${price:.3f}/query, 1M q/day, 69% deflect: "
              f"save ${saved_day*365:,.0f}/year")

    print("\nNOTE: ranges, not guarantees. CPU energy is attributed/estimated;")
    print("LLM per-query energy and volume are vendor/IEA figures; deflection")
    print("is from our synthetic benchmark. See docs/PAPER.md for caveats.")


if __name__ == "__main__":
    main()
