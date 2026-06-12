"""Shadow-mode replay harness: GeoCell's go-to-market instrument.

Point this at a customer's exported corpus and a sample of their real
query log. It runs entirely offline and read-only -- it never calls an
LLM and never touches their production system -- and emits a report
answering the only questions a buyer cares about:

  - What fraction of your queries could be answered from memory, for free,
    with a citation? (deflection)
  - Of those, how often does GeoCell agree with the answer your current
    system already gave? (agreement-with-incumbent)
  - Where does GeoCell DISAGREE -- i.e. where did it find a contradiction,
    a superseded fact, or a stale answer your system got wrong? (the
    review list: the "we found N issues in your own data" slide)
  - How much model spend and energy would deflection save at your volume?

Input formats (JSON Lines, trivial to export):

  corpus.jsonl   one document per line:
    {"id": "...", "text": "...", "source": "...", "date": "YYYY-MM-DD",
     "authority": 0.0-1.0, "confidence": 0.0-1.0}

  queries.jsonl  one query per line:
    {"query": "...", "answer": "<your system's logged answer>" (optional),
     "subject": "<entity hint, optional>"}

Run:  python -m geocell.shadow corpus.jsonl queries.jsonl --out report.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field as dfield
from typing import Any, Dict, List, Optional

from geocell.field import GeoCellField

# Defaults are conservative, published anchors; override at the CLI to use
# the customer's own pricing/energy figures.
DEFAULT_USD_PER_QUERY = 0.005
DEFAULT_WH_PER_QUERY = 0.34          # OpenAI "average" query
GEOCELL_WH_PER_QUERY = 0.0004        # ~50 ms CPU, attributed

_NUMERIC_Q = {"cost", "charge", "charges", "budget", "much", "price", "value",
              "amount", "rate", "revenue", "annual", "fee", "many"}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _agree(a: str, b: str) -> bool:
    """Do two answers state the same fact? When both carry numbers, the
    numbers must match (a budget of $2M and $2.7M disagree even though both
    say 'Project Orion'); otherwise fall back to proper-noun / substring
    overlap. Number-aware comparison is what makes the disagreement list
    trustworthy."""
    if not a or not b:
        return False
    from geocell.text import extract_claim_values
    va = {round(float(x["value"]), 4) for x in extract_claim_values(a) if x["value"] is not None}
    vb = {round(float(x["value"]), 4) for x in extract_claim_values(b) if x["value"] is not None}
    if va and vb:
        return bool(va & vb)
    if va or vb:
        return False                       # one cites a number, the other doesn't
    na = set(re.findall(r"\b[A-Z][a-zA-Z]+\b", a))
    nb = set(re.findall(r"\b[A-Z][a-zA-Z]+\b", b))
    if na and nb:
        return len(na & nb) / max(1, min(len(na), len(nb))) >= 0.5
    al, bl = a.lower(), b.lower()
    return al in bl or bl in al


@dataclass
class ShadowConfig:
    resonance_threshold: float = 0.45
    margin_threshold: float = 0.08
    strong_resonance: float = 0.9
    min_vocab_coverage: float = 0.4
    usd_per_query: float = DEFAULT_USD_PER_QUERY
    wh_per_query: float = DEFAULT_WH_PER_QUERY


@dataclass
class ShadowReport:
    queries: int = 0
    deflected: int = 0
    agreements: int = 0
    disagreements: List[Dict[str, Any]] = dfield(default_factory=list)
    deflected_rows: List[Dict[str, Any]] = dfield(default_factory=list)
    escalated: List[str] = dfield(default_factory=list)
    corpus_contradictions: List[Dict[str, Any]] = dfield(default_factory=list)
    field_stats: Dict[str, Any] = dfield(default_factory=dict)
    config: Dict[str, Any] = dfield(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        n = max(1, self.queries)
        defl = self.deflected
        usd = self.config.get("usd_per_query", DEFAULT_USD_PER_QUERY)
        wh = self.config.get("wh_per_query", DEFAULT_WH_PER_QUERY)
        with_log = self.agreements + len(self.disagreements)
        return {
            "queries_replayed": self.queries,
            "deflected": defl,
            "deflection_rate": round(defl / n, 3),
            "escalated_to_llm": len(self.escalated),
            "agreement_with_incumbent": f"{self.agreements}/{with_log}" if with_log else "n/a",
            "agreement_rate": round(self.agreements / max(1, with_log), 3) if with_log else None,
            "disagreements_to_review": len(self.disagreements),
            "corpus_contradictions_found": len(self.corpus_contradictions),
            "projected_annual_usd_saved_at_replayed_volume": round(defl * usd, 4),
            "projected_annual_wh_saved_at_replayed_volume": round(defl * (wh - GEOCELL_WH_PER_QUERY), 4),
            "note": "Savings shown are for the replayed sample. Multiply by "
                    "(annual_volume / sample_size) for the annual figure.",
        }


def load_corpus(field: GeoCellField, docs: List[Dict[str, Any]]) -> None:
    for d in docs:
        for sent in _split_sentences(d.get("text", "")):
            field.ingest(
                sent,
                source=str(d.get("source") or d.get("id") or "doc"),
                date=str(d.get("date", "")),
                authority=float(d.get("authority", 0.5)),
                confidence=float(d.get("confidence", 0.75)),
                subject=d.get("subject"),
            )


def _served(field: GeoCellField, query: str, cfg: ShadowConfig):
    """Return (answer_dict, served:bool) applying confident abstention:
    serve only on a clear, on-topic winner."""
    from geocell.text import stem, tokenize
    # Out-of-domain gate: if most of the query's content words never appear
    # in the corpus, the question is about something we don't hold -- a
    # lone match on a common word ("policy") must not trigger a confident
    # answer. Escalate instead.
    qterms = {stem(t) for t in tokenize(query)}
    if qterms:
        known = sum(1 for t in qterms if t in field._postings)
        if known / len(qterms) < cfg.min_vocab_coverage:
            return field.ask(query), False
    ans = field.ask(query)
    ev = ans.get("evidence", [])
    if not ev:
        return ans, False
    top = ev[0]
    served = top["resonance"] >= cfg.resonance_threshold
    if served and len(ev) >= 2:
        margin = top["resonance"] - ev[1]["resonance"]
        if margin < cfg.margin_threshold and top["resonance"] < cfg.strong_resonance:
            served = False
    if served and set(query.lower().replace("?", "").split()) & _NUMERIC_Q:
        if not re.search(r"\d", ans["answer"]):
            served = False
    return ans, served


def run_shadow(docs: List[Dict[str, Any]], queries: List[Dict[str, Any]],
               cfg: Optional[ShadowConfig] = None) -> ShadowReport:
    cfg = cfg or ShadowConfig()
    field = GeoCellField()
    load_corpus(field, docs)
    field.hypothesize()
    field.propagate_trust()

    rep = ShadowReport(config=cfg.__dict__.copy())
    for q in queries:
        rep.queries += 1
        query = q["query"]
        logged = q.get("answer")
        ans, served = _served(field, query, cfg)
        if not served:
            rep.escalated.append(query)
            continue
        rep.deflected += 1
        cell = field.cells[ans["evidence"][0]["id"]]
        row = {
            "query": query,
            "geocell_answer": ans["answer"],
            "citation": cell.source,
            "belief_status": ans.get("belief_status"),
            "trust": ans.get("trust"),
        }
        rep.deflected_rows.append(row)
        if logged is not None:
            if _agree(ans["answer"], logged):
                rep.agreements += 1
            else:
                rep.disagreements.append({
                    **row,
                    "incumbent_answer": logged,
                    "why": ("memory holds a superseded/contested fact"
                            if ans.get("belief_status") in ("superseded", "contested")
                            or ans.get("superseded_history") or ans.get("open_contradictions")
                            else "different fact retrieved -- review"),
                })

    # Corpus-level contradictions (the audit surface), deduped by pair.
    seen = set()
    for c in field.contradictions(include_resolved=True):
        key = tuple(sorted((c["a_id"], c["b_id"])))
        if key in seen:
            continue
        seen.add(key)
        rep.corpus_contradictions.append({
            "a": c["a"], "a_source": c["a_source"], "a_status": c["a_status"],
            "b": c["b"], "b_source": c["b_source"], "b_status": c["b_status"],
            "reason": c["reason"], "resolved": c["resolved"],
        })
    rep.field_stats = field.stats()
    return rep


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def print_report(rep: ShadowReport) -> None:
    s = rep.summary()
    print("=" * 64)
    print("GeoCell Shadow-Mode Report (offline, read-only, 0 LLM calls)")
    print("=" * 64)
    print(f"  corpus: {rep.field_stats.get('cells')} claims, "
          f"{rep.field_stats.get('edges')} relations, "
          f"{rep.field_stats.get('open_contradictions')} open contradictions")
    print(f"  queries replayed:        {s['queries_replayed']}")
    print(f"  deflected (no LLM):      {s['deflected']}  ({s['deflection_rate']:.0%})")
    print(f"  agreement w/ incumbent:  {s['agreement_with_incumbent']}")
    print(f"  disagreements to review: {s['disagreements_to_review']}")
    print(f"  corpus contradictions:   {s['corpus_contradictions_found']}")
    print(f"  $ saved (sample):        ${s['projected_annual_usd_saved_at_replayed_volume']}")
    print(f"  Wh saved (sample):       {s['projected_annual_wh_saved_at_replayed_volume']}")
    if rep.disagreements:
        print("\n  --- DISAGREEMENTS (review these: stale/contradicted answers) ---")
        for d in rep.disagreements[:10]:
            print(f"   Q: {d['query']}")
            print(f"     GeoCell : {d['geocell_answer']}  [{d['citation']}] "
                  f"({d['belief_status']})")
            print(f"     Yours   : {d['incumbent_answer']}")
            print(f"     why     : {d['why']}")
    if rep.corpus_contradictions:
        print("\n  --- CONTRADICTIONS FOUND IN YOUR CORPUS ---")
        for c in rep.corpus_contradictions[:10]:
            print(f"   [{c['a_status']}] {c['a'][:60]}  ({c['a_source']})")
            print(f"      vs [{c['b_status']}] {c['b'][:60]}  ({c['b_source']})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="GeoCell shadow-mode replay harness")
    ap.add_argument("corpus", help="corpus JSONL")
    ap.add_argument("queries", help="query-log JSONL")
    ap.add_argument("--out", help="write full report JSON here")
    ap.add_argument("--usd-per-query", type=float, default=DEFAULT_USD_PER_QUERY)
    ap.add_argument("--wh-per-query", type=float, default=DEFAULT_WH_PER_QUERY)
    args = ap.parse_args(argv)

    cfg = ShadowConfig(usd_per_query=args.usd_per_query, wh_per_query=args.wh_per_query)
    rep = run_shadow(_read_jsonl(args.corpus), _read_jsonl(args.queries), cfg)
    print_report(rep)
    if args.out:
        payload = {"summary": rep.summary(), "disagreements": rep.disagreements,
                   "corpus_contradictions": rep.corpus_contradictions,
                   "deflected": rep.deflected_rows, "field_stats": rep.field_stats}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  full report -> {args.out}")


if __name__ == "__main__":
    main()
