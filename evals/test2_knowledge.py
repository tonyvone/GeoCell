"""Test 2 — Verified Briefing / Hallucination Quarantine.

Loads the enterprise corpus into a GeoCell field, runs belief revision +
inference + trust propagation, then answers 100 gold questions two ways:

  baseline : the LLM alone (no memory)
  geooracle: GeoCell first, LLM only on a miss, with hallucinated LLM
             answers quarantined when they contradict trusted memory.

Scores citation accuracy, contradiction-trap detection, LLM calls
avoided, answer consistency (stability across paraphrases), latency and
cost.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import build_corpus
from llm_client import LLMClient, PRICE_OUT

from geocell import GeoCellField

LLM = LLMClient(model="haiku")


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def load_field(corpus) -> GeoCellField:
    """Ingest at sentence granularity. Real documents bundle several
    facts; chunking lets a budget revision supersede only the budget
    claim, not the whole charter (owner, department stay intact)."""
    field = GeoCellField()
    for d in corpus.docs:
        for sent in split_sentences(d.text):
            field.ingest(sent, source=d.doc_id, date=d.date,
                         confidence=d.confidence, authority=d.authority)
    field.hypothesize()
    field.propagate_trust()
    return field


def answer_matches(text: str, gold: str) -> bool:
    t = text.lower().replace(",", "")
    g = gold.lower().replace(",", "")
    return g in t


# --------------------------------------------------------------------------
# Baseline: LLM with no memory (closed-book over the same corpus facts)
# --------------------------------------------------------------------------

def baseline_answer(corpus_index: str, question: str) -> str:
    prompt = (
        "Answer the question in one short sentence using ONLY the facts "
        "below. If the answer is not present, say 'unknown'.\n\n"
        f"FACTS:\n{corpus_index}\n\nQUESTION: {question}\nANSWER:"
    )
    return LLM.complete(prompt)


def corpus_index_text(corpus, limit_chars: int = 9000) -> str:
    # A compact fact sheet the baseline LLM may read (closed-book RAG-style).
    lines = [f"[{d.doc_id}] {d.text}" for d in corpus.docs]
    txt, out = 0, []
    for ln in lines:
        if txt + len(ln) > limit_chars:
            break
        out.append(ln); txt += len(ln)
    return "\n".join(out)


# --------------------------------------------------------------------------
# GeoOracle answering with citation + contradiction handling
# --------------------------------------------------------------------------

_NUMERIC_Q = {"cost", "charge", "charges", "budget", "much", "price", "value", "annual", "fee"}


def geocell_answer(field: GeoCellField, question: str):
    ans = field.ask(question)
    ev = ans.get("evidence", [])
    top = ev[0] if ev else None
    resonant = bool(top) and top["resonance"] >= 0.45
    # Confidence by margin: only serve when the top candidate clearly beats
    # the runner-up. A near-tie means the discriminating term didn't land
    # (the conversational paraphrases), so defer to the LLM rather than
    # guess between siblings. This is selective prediction / abstention.
    if resonant and len(ev) >= 2:
        margin = top["resonance"] - ev[1]["resonance"]
        if margin < 0.08 and top["resonance"] < 0.9:
            resonant = False
    # Answerability gate: a question asking for a quantity is only served
    # from memory when the answer actually carries a number.
    if resonant and set(question.lower().replace("?", "").split()) & _NUMERIC_Q:
        if not re.search(r"\d", ans["answer"]):
            resonant = False
    cite = top["source"] if top else None
    return {
        "answer": ans["answer"],
        "belief_status": ans.get("belief_status"),
        "trust": ans.get("trust"),
        "resonance": top["resonance"] if top else 0.0,
        "citation": cite,
        "open_contradictions": ans.get("open_contradictions", []),
        "resonant": resonant,
    }


def _predicate_overlap(a_text: str, b_text: str, subject: str) -> float:
    """Shared non-subject, non-numeric content words. Distinguishes a real
    contradiction (same predicate, different value) from two unrelated
    quantities about the same subject (a budget vs a vendor cost)."""
    from geocell.text import tokenize
    sub = set(subject.split())
    keep = lambda s: {t for t in tokenize(s)
                      if not t.replace("$", "").replace(".", "").replace(",", "").isdigit()} - sub
    ta, tb = keep(a_text), keep(b_text)
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def adjudicate_llm(field: GeoCellField, llm_text: str, question: str):
    """Non-mutating hallucination check: does the LLM's answer conflict
    with a trusted memory claim on the *same subject and same predicate*?
    If so, quarantine it and return the trusted belief instead. Mirrors
    the field's supersession rule (including its predicate-overlap guard)
    without polluting the live field."""
    from geocell.text import extract_claim_values, extract_subject
    subj = extract_subject(llm_text)
    vals = {round(float(v["value"] or 0), 4) for v in extract_claim_values(llm_text) if v["value"]}
    if not vals:
        return False, None
    for c in field.cells:
        if c.status != "active" or c.kind != "observed" or not c.values:
            continue
        if field._subject_overlap(c.subject, subj) < 0.99:
            continue
        cvals = {round(float(v["value"] or 0), 4) for v in c.values if v["value"]}
        if cvals and not (cvals & vals) and c.trust >= 0.6 \
                and _predicate_overlap(llm_text, c.content, c.subject) >= 0.3:
            return True, c.content   # same predicate, different value -> quarantine
    return False, None


def run_geooracle(field, corpus, idx):
    rows = []
    llm_calls = quarantined = 0
    for qa in corpus.qa:
        t0 = time.perf_counter()
        g = geocell_answer(field, qa.question)
        geo_ms = (time.perf_counter() - t0) * 1000

        row = {"qid": qa.qid, "qtype": qa.qtype, "gold": qa.answer}
        if g["resonant"]:
            row.update(origin="geocell", answer=g["answer"], citation=g["citation"],
                       belief_status=g["belief_status"], geo_ms=round(geo_ms, 2),
                       correct=answer_matches(g["answer"], qa.answer),
                       trap_handled=(qa.qtype == "contradiction" and answer_matches(g["answer"], qa.answer)))
        else:
            text = baseline_answer(idx, qa.question)
            llm_calls += 1
            q, trusted = adjudicate_llm(field, text, qa.question)
            if q:
                quarantined += 1
                row.update(origin="field_override", answer=trusted, citation="memory",
                           quarantined_llm=text, geo_ms=round(geo_ms, 2),
                           correct=answer_matches(trusted, qa.answer))
            else:
                row.update(origin="llm", answer=text, citation=None,
                           geo_ms=round(geo_ms, 2), correct=answer_matches(text, qa.answer))
        rows.append(row)
    return rows, llm_calls, quarantined


def hallucination_probes(field, corpus):
    """Fabricated LLM answers that contradict trusted budget memory. The
    adjudicator must quarantine each and recover the trusted value."""
    caught = 0
    details = []
    for proj in ["Orion", "Helios", "Vega", "Atlas", "Nova"]:
        fake = f"The Project {proj} budget is $99,000,000."
        q, trusted = adjudicate_llm(field, fake, f"Project {proj} budget?")
        caught += int(q)
        details.append({"probe": fake, "quarantined": q,
                        "recovered": trusted if q else None})
    return {"caught": caught, "total": len(details), "details": details}


def run_baseline(corpus, idx):
    rows = []
    for qa in corpus.qa:
        text = baseline_answer(idx, qa.question)
        rows.append({"qid": qa.qid, "qtype": qa.qtype, "gold": qa.answer,
                     "answer": text, "correct": answer_matches(text, qa.answer),
                     "trap_handled": (qa.qtype == "contradiction" and answer_matches(text, qa.answer))})
    return rows


def consistency(rows, corpus):
    """Fraction of paraphrase groups answered identically. Near-repeat
    questions about the same project owner should yield one stable answer."""
    groups = {}
    for qa, r in zip(corpus.qa, rows):
        if qa.qtype == "nearrepeat":
            groups.setdefault(qa.answer, []).append(r["answer"].strip().lower())
    stable = sum(1 for v in groups.values() if len(set(v)) == 1)
    return round(stable / max(1, len(groups)), 3)


def score(rows, label):
    by = {}
    for r in rows:
        by.setdefault(r["qtype"], [0, 0])
        by[r["qtype"]][0] += int(r["correct"]); by[r["qtype"]][1] += 1
    acc = {k: f"{v[0]}/{v[1]}" for k, v in by.items()}
    total = sum(v[0] for v in by.values())
    return {"label": label, "accuracy_total": f"{total}/{len(rows)}",
            "accuracy_by_type": acc}


def main():
    corpus = build_corpus()
    idx = corpus_index_text(corpus)
    field = load_field(corpus)
    print(f"[corpus] {len(corpus.docs)} docs, {len(corpus.qa)} questions; "
          f"field={len(field.cells)} cells")

    base_rows = run_baseline(corpus, idx)
    base_score = score(base_rows, "llm_only")

    geo_rows, llm_calls, quarantined = run_geooracle(field, corpus, idx)
    geo_score = score(geo_rows, "geooracle")
    probes = hallucination_probes(field, corpus)

    cited = [r for r in geo_rows if r.get("citation")]
    cite_correct = sum(1 for r in cited if r["correct"])
    geocell_served = [r for r in geo_rows if r["origin"] in ("geocell", "field_override")]

    traps = [r for r in geo_rows if r["qtype"] == "contradiction"]
    traps_base = [r for r in base_rows if r["qtype"] == "contradiction"]
    trap_caught = sum(1 for r in traps if r["trap_handled"])
    trap_caught_base = sum(1 for r in traps_base if r["trap_handled"])

    summary = {
        "baseline": base_score,
        "geooracle": geo_score,
        "llm_calls_baseline": len(corpus.qa),
        "llm_calls_geooracle": llm_calls,
        "llm_calls_avoided": len(corpus.qa) - llm_calls,
        "call_avoidance_rate": round((len(corpus.qa) - llm_calls) / len(corpus.qa), 3),
        "citation_accuracy": f"{cite_correct}/{len(cited)}" if cited else "0/0",
        "citation_accuracy_rate": round(cite_correct / max(1, len(cited)), 3),
        "contradiction_traps_caught_geooracle": f"{trap_caught}/{len(traps)}",
        "contradiction_traps_caught_baseline": f"{trap_caught_base}/{len(traps_base)}",
        "hallucinations_quarantined_live": quarantined,
        "hallucination_probes": f"{probes['caught']}/{probes['total']}",
        "consistency_geooracle": consistency(geo_rows, corpus),
        "consistency_baseline": consistency(base_rows, corpus),
        "llm_report": LLM.report(),
    }

    print("\n=== TEST 2 RESULTS ===")
    print(json.dumps(summary, indent=2))
    print("\n--- hallucination probes ---")
    print(json.dumps(probes, indent=2))
    json.dump({"summary": summary, "probes": probes, "geo_rows": geo_rows, "base_rows": base_rows},
              open("evals/test2_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
