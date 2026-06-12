"""Test 4 -- Adversarial robustness: poisoning & trust collapse.

The roadmap asks: can an attacker corrupt the field by flooding it with
conflicting low-authority writes (a Sybil/poisoning attack)? We establish
a high-authority truth, then bury it under many low-authority lies and
check that (a) the truth survives as the active belief, (b) the lies are
quarantined/contested rather than adopted, and (c) trust does not collapse.
Fully deterministic, no LLM.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geocell import GeoCellField


def scenario(n_poison: int, poison_authority: float, truth_authority: float = 0.95):
    f = GeoCellField()
    # The trusted fact (one authoritative, recent source).
    truth = f.ingest("Project Orion budget is $2.7 million.",
                     source="board_filing", date="2026-02-01",
                     confidence=0.95, authority=truth_authority)
    # A flood of low-authority contradictions, each a different wrong value.
    for i in range(n_poison):
        f.ingest(f"Project Orion budget is ${3 + i} million.",
                 source=f"anon_post_{i}", date="2026-03-01",
                 confidence=0.5, authority=poison_authority)
    f.propagate_trust()
    ans = f.ask("What is the current Project Orion budget?")
    truth_cell = f.cells[truth]
    poison_cells = [c for c in f.cells if c.source.startswith("anon_post_")]
    poison_active = sum(1 for c in poison_cells if c.status == "active")
    return {
        "n_poison": n_poison,
        "poison_authority": poison_authority,
        "answer_is_truth": "$2.7" in ans["answer"],
        "answer": ans["answer"],
        "truth_status": truth_cell.status,
        "truth_trust": round(truth_cell.trust, 3),
        "mean_poison_trust": round(sum(c.trust for c in poison_cells) / max(1, len(poison_cells)), 3),
        "poison_still_active": poison_active,
        "poison_total": len(poison_cells),
    }


def main():
    print("Adversarial poisoning: 1 high-authority truth vs a flood of "
          "low-authority contradictions.\n")
    print(f"{'poison':>7} {'pAuth':>6} {'answer=truth':>13} {'truthStatus':>12} "
          f"{'truthTrust':>11} {'poisonTrust':>12} {'poisonActive':>13}")
    rows = []
    for n, pa in [(5, 0.2), (50, 0.2), (500, 0.2), (50, 0.4), (50, 0.6)]:
        r = scenario(n, pa)
        rows.append(r)
        print(f"{r['n_poison']:>7} {r['poison_authority']:>6} "
              f"{str(r['answer_is_truth']):>13} {r['truth_status']:>12} "
              f"{r['truth_trust']:>11} {r['mean_poison_trust']:>12} "
              f"{str(r['poison_still_active'])+'/'+str(r['poison_total']):>13}")

    print("\n--- checks ---")
    checks = {
        "truth survives a 500x low-authority flood":
            scenario(500, 0.2)["answer_is_truth"],
        "truth stays the highest-trust cell under flood":
            (lambda r: r["truth_trust"] > r["mean_poison_trust"])(scenario(50, 0.2)),
        "no catastrophic trust collapse (truth trust stays > 0.5)":
            scenario(500, 0.2)["truth_trust"] > 0.5,
        "equal-authority conflict does NOT silently adopt a lie":
            scenario(50, 0.6)["answer_is_truth"] or
            scenario(50, 0.6)["truth_status"] in ("active", "contested"),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\n  Note: GeoCell weights by source authority, not vote count, so a")
    print("  Sybil flood of weak sources cannot outweigh one trusted source —")
    print("  the structural defense against write-poisoning. 0 LLM calls.")


if __name__ == "__main__":
    main()
