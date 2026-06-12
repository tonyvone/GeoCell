import json
import os

from geocell.shadow import ShadowConfig, run_shadow, _agree

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "evals", "shadow_sample")


def _read(name):
    out = []
    with open(os.path.join(SAMPLE, name)) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def test_agreement_is_number_aware():
    # same number -> agree, even with extra words
    assert _agree("$1,500.", "The deductible is $1,500 for individuals.")
    # different numbers -> disagree, even if proper nouns overlap
    assert not _agree("Project Orion budget is $2 million.",
                      "The Project Orion budget was revised to $2.7 million.")
    # one cites a number, the other doesn't -> disagree
    assert not _agree("18 days", "a generous allowance")


def test_shadow_report_surfaces_stale_answers():
    rep = run_shadow(_read("corpus.jsonl"), _read("queries.jsonl"))
    s = rep.summary()
    # deflects most of the answerable queries, escalates the unknown one
    assert rep.deflected >= 8
    assert s["deflection_rate"] >= 0.8
    assert any("generative AI" in q for q in rep.escalated)

    # the three planted revisions must show up as disagreements with the
    # incumbent's logged (stale) answers
    dq = " | ".join(d["query"] for d in rep.disagreements)
    assert "vacation" in dq and "remotely" in dq and "Orion budget" in dq
    assert len(rep.disagreements) >= 3
    # every disagreement carries a citation and a belief status
    for d in rep.disagreements:
        assert d["citation"] and d["belief_status"]

    # and the corpus audit surfaces the underlying contradictions
    assert s["corpus_contradictions_found"] >= 3


def test_shadow_is_offline_and_deterministic():
    docs, qs = _read("corpus.jsonl"), _read("queries.jsonl")
    a = run_shadow(docs, qs).summary()
    b = run_shadow(docs, qs).summary()
    assert a == b                      # deterministic, no LLM, no randomness


def test_savings_scale_with_pricing():
    docs, qs = _read("corpus.jsonl"), _read("queries.jsonl")
    cheap = run_shadow(docs, qs, ShadowConfig(usd_per_query=0.001)).summary()
    dear = run_shadow(docs, qs, ShadowConfig(usd_per_query=0.02)).summary()
    assert dear["projected_annual_usd_saved_at_replayed_volume"] > \
        cheap["projected_annual_usd_saved_at_replayed_volume"]
