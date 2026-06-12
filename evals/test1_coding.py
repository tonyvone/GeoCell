"""Test 1 — Repeat Bug / Verified Fix.

Train: an LLM authors a fix for some bugs; only fixes the real test suite
accepts are distilled into GeoCell as generalized transformations.
Test:  near-repeat bugs (sibling functions) are fixed by GeoCell FIRST,
       with no LLM call, and only escalated to the LLM on a miss.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict

from bugs import BUGS, by_group
from fix_memory import FixMemory, validate_patch
from harness_util import (
    classify_failure, fresh_repo, inject, read_func, replace_func, run_full_suite, run_test,
)
from llm_client import LLMClient, PRICE_OUT

LLM = LLMClient(model="haiku")


def split_groups():
    train, test = [], []
    for members in by_group().values():
        k = math.ceil(len(members) / 2)
        train.extend(members[:k])
        test.extend(members[k:])
    return train, test


def strip_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    body = m.group(1) if m else text
    return body.strip("\n")


def llm_fix_function(buggy_src: str, test_log: str, func: str) -> str:
    prompt = (
        "You are fixing a Python bug. Return ONLY the corrected function "
        "definition, no explanation, no markdown fences.\n\n"
        f"This function `{func}` is buggy:\n\n{buggy_src}\n\n"
        f"Its test fails:\n{test_log[-600:]}\n\n"
        f"Return the complete corrected `def {func}` with minimal changes."
    )
    return strip_code(LLM.complete(prompt))


def train_phase():
    mem = FixMemory()
    learned = 0
    for bug in TRAIN:
        repo = fresh_repo()
        if not inject(repo, bug.file, bug.broken, bug.clean, bug.func):
            continue
        ok, log = run_test(repo, bug.test)
        if ok:
            continue  # injection didn't actually break the test; skip
        buggy_src = read_func(repo, bug.file, bug.func)
        fixed_src = llm_fix_function(buggy_src, log, bug.func)
        if not replace_func(repo, bug.file, bug.func, fixed_src):
            continue
        ok, _ = run_test(repo, bug.test)
        if not ok:
            continue  # the LLM's own fix didn't pass; nothing trustworthy to learn
        idents = [bug.func] + bug.params
        if mem.learn(bug.symptom, buggy_src, fixed_src, idents, bug.id):
            learned += 1
    return mem, learned


def test_phase(mem: FixMemory):
    rows = []
    for bug in TEST:
        repo = fresh_repo()
        inject(repo, bug.file, bug.broken, bug.clean, bug.func)
        ok, log = run_test(repo, bug.test)
        if ok:
            continue
        buggy_src = read_func(repo, bug.file, bug.func)
        file_text = (repo / bug.file).read_text()

        row = {"bug": bug.id, "group": bug.group, "category": bug.category}

        t0 = time.perf_counter()
        proposal = mem.propose(bug.symptom, buggy_src, full_text=file_text)
        row["geocell_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if proposal and proposal.get("quarantined"):
            row.update(origin="llm_after_quarantine", quarantined_geocell=True,
                       quarantine_reason=proposal["reason"])
            proposal = None

        if proposal and proposal.get("patched"):
            replace_func(repo, bug.file, bug.func, proposal["patched"])
            ok, log = run_test(repo, bug.test)
            if ok:
                row.update(origin="geocell", fixed=True, llm_call=False,
                           via=proposal["pattern_bug"], resonance=round(proposal["resonance"], 3))
                rows.append(row); continue
            # geometric fix didn't verify: revert and fall back
            (repo / bug.file).write_text(file_text)
            row["geocell_proposed_but_failed"] = True

        # LLM fallback (with the same safety gate)
        t0 = time.perf_counter()
        fixed_src = llm_fix_function(buggy_src, log, bug.func)
        row["llm_s"] = round(time.perf_counter() - t0, 2)
        replace_func(repo, bug.file, bug.func, fixed_src)
        patched = (repo / bug.file).read_text()
        gate_ok, reason = validate_patch(patched)
        if not gate_ok:
            row.update(origin="llm", fixed=False, llm_call=True,
                       quarantined_llm=True, quarantine_reason=reason)
            rows.append(row); continue
        ok, _ = run_test(repo, bug.test)
        row.update(origin=row.get("origin", "llm"), fixed=ok,
                   llm_call=True)
        rows.append(row)
    return rows


def baseline_llm_only():
    """Every test bug fixed by the LLM, no memory. Cost/latency reference."""
    total_s, fixed = 0.0, 0
    for bug in TEST:
        repo = fresh_repo()
        inject(repo, bug.file, bug.broken, bug.clean, bug.func)
        ok, log = run_test(repo, bug.test)
        if ok:
            continue
        buggy_src = read_func(repo, bug.file, bug.func)
        t0 = time.perf_counter()
        fixed_src = llm_fix_function(buggy_src, log, bug.func)
        total_s += time.perf_counter() - t0
        replace_func(repo, bug.file, bug.func, fixed_src)
        ok, _ = run_test(repo, bug.test)
        fixed += int(ok)
    return {"fixed": fixed, "total": len(TEST), "wall_s": round(total_s, 1)}


def hallucination_probes(mem: FixMemory):
    """Adversarial patches a careless agent might emit. The safety gate
    must quarantine each before it can reach the repo."""
    probes = {
        "fake_stdlib_attr": "import hmac\ndef verify(a, b):\n    return hmac.safe_equals(a, b)\n",
        "invented_module": "import suprsecure\ndef verify(a, b):\n    return suprsecure.check(a, b)\n",
        "undefined_helper_import": "from utils import constant_time_eq\ndef verify(a, b):\n    return constant_time_eq(a, b)\n",
        "fake_hashlib_algo": "import hashlib\ndef ck(d):\n    return hashlib.sha9000(d).hexdigest()\n",
        "syntax_error": "def broken(x)\n    return x\n",
    }
    caught = {}
    for name, src in probes.items():
        ok, reason = validate_patch(src)
        caught[name] = {"quarantined": not ok, "reason": reason}
    return caught


def main():
    global TRAIN, TEST
    TRAIN, TEST = split_groups()
    print(f"[split] {len(TRAIN)} train bugs, {len(TEST)} near-repeat test bugs")

    t0 = time.perf_counter()
    mem, learned = train_phase()
    print(f"[train] learned {learned} generalized fixes from {len(TRAIN)} bugs "
          f"({round(time.perf_counter()-t0,1)}s)")

    rows = test_phase(mem)
    probes = hallucination_probes(mem)
    base = baseline_llm_only()

    fixed_rows = [r for r in rows if r.get("fixed")]
    geocell_fixed = [r for r in rows if r.get("origin") == "geocell" and r.get("fixed")]
    llm_fixed = [r for r in fixed_rows if r.get("llm_call")]
    geocell_q = [r for r in rows if r.get("quarantined_geocell")]
    llm_q = [r for r in rows if r.get("quarantined_llm")]
    n = len(rows)

    geo_ms = [r["geocell_ms"] for r in rows if "geocell_ms" in r]
    llm_lat = [r["llm_s"] for r in rows if "llm_s" in r]
    avg_geo = round(sum(geo_ms) / max(1, len(geo_ms)), 2)
    avg_llm = round(sum(llm_lat) / max(1, len(llm_lat)), 2) if llm_lat else None

    llm_calls_made = sum(1 for r in rows if r.get("llm_call"))
    calls_avoided = len(geocell_fixed)
    cost_per = LLM.out_tokens / max(1, LLM.calls) / 1e6 * PRICE_OUT  # avg $/call estimate

    summary = {
        "test_bugs": n,
        "fixed_by_geocell_no_llm": len(geocell_fixed),
        "geocell_no_llm_rate": round(len(geocell_fixed) / max(1, n), 3),
        "fixed_by_llm_fallback": len(llm_fixed),
        "total_fixed": len(fixed_rows),
        "overall_fix_rate": round(len(fixed_rows) / max(1, n), 3),
        "llm_calls_in_test_phase": llm_calls_made,
        "llm_calls_avoided_by_geocell": calls_avoided,
        "call_reduction_vs_llm_only": round(calls_avoided / max(1, n), 3),
        "avg_geocell_latency_ms": avg_geo,
        "avg_llm_latency_s": avg_llm,
        "speedup_x": round((avg_llm * 1000 / avg_geo), 1) if avg_llm and avg_geo else None,
        "geocell_quarantines": len(geocell_q),
        "llm_fix_quarantines": len(llm_q),
        "hallucination_probes_caught": sum(1 for v in probes.values() if v["quarantined"]),
        "hallucination_probes_total": len(probes),
        "baseline_llm_only": base,
        "llm_report": LLM.report(),
    }

    # Full-suite pass rate after applying GeoCell fixes to a multi-bug repo.
    summary["post_fix_suite"] = multi_bug_suite(mem)

    print("\n=== TEST 1 RESULTS ===")
    print(json.dumps(summary, indent=2))
    print("\n--- hallucination gate ---")
    print(json.dumps(probes, indent=2))
    print("\n--- per-bug ---")
    for r in rows:
        print(f"  {r['bug']:<14} {r.get('origin','?'):<22} "
              f"fixed={r.get('fixed')} via={r.get('via','-')} res={r.get('resonance','-')}")
    json.dump({"summary": summary, "rows": rows, "probes": probes},
              open("evals/test1_results.json", "w"), indent=2)


def multi_bug_suite(mem: FixMemory):
    """Inject ALL near-repeat test bugs into one repo, let GeoCell patch
    every one it can without an LLM, then report the suite pass rate."""
    repo = fresh_repo()
    for bug in TEST:
        inject(repo, bug.file, bug.broken, bug.clean, bug.func)
    before = run_full_suite(repo)
    patched_any = True
    rounds = 0
    while patched_any and rounds < 5:
        patched_any = False
        rounds += 1
        for bug in TEST:
            file_text = (repo / bug.file).read_text()
            buggy_src = read_func(repo, bug.file, bug.func)
            if buggy_src is None:
                continue
            prop = mem.propose(bug.symptom, buggy_src, full_text=file_text)
            if prop and prop.get("patched"):
                replace_func(repo, bug.file, bug.func, prop["patched"])
                patched_any = True
    after = run_full_suite(repo)
    return {"before_fix": {"passed": before[0], "failed": before[1]},
            "after_geocell_fixes": {"passed": after[0], "failed": after[1]}}


if __name__ == "__main__":
    main()
