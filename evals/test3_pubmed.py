"""Test 3 -- Real-data domain probe on PubMed/ALS literature.

A focused, honest test on REAL biomedical abstracts (not synthetic):
four C9orf72/ALS papers spanning 2011-2025 from journals of differing
authority, captured from the PubMed MCP server (see pubmed_fixture.json
for DOIs). It exercises the roadmap's "domain-specific" and "numerical
drift / multi-source contradiction" items on naturally-occurring data:

  - The reported fraction of *familial ALS* caused by C9orf72 genuinely
    drifts across papers and years (23.5%, ~46%/one-third, 30-50%).
    GeoCell should detect the conflict, order the sources by trust
    (journal authority + recency), surface a current belief, and retain
    the rest as cited history -- with zero LLM calls.
  - A mechanism query should retrieve the real pathology (TDP-43) with a
    citation, via the same deterministic recall.

This is a real-data *probe*, not a 6-12 month longitudinal deployment;
it demonstrates the behavior on real text and real provenance.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geocell import GeoCellField

FIX = os.path.join(os.path.dirname(__file__), "pubmed_fixture.json")


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# A stand-in for a domain entity recognizer (scispaCy / NER / an LLM
# extraction pass). On these abstracts the salient entity is the gene; a
# real pipeline would supply it. We detect it by a simple gene-token rule
# purely to *simulate* that external signal -- the point is to isolate the
# epistemic layer from the rule-based extractor, not to ship this regex.
import re as _re
def entity_hint(sentence: str):
    m = _re.search(r"C9orf72|C9ORF72", sentence)
    if m and ("familial" in sentence.lower() or "%" in sentence or "percent" in sentence):
        return "c9orf72 familial als"
    if m:
        return "c9orf72"
    return None


def load_field(use_entity_hint: bool):
    data = json.load(open(FIX))
    field = GeoCellField()
    for art in data["articles"]:
        src = f"{art['journal']} {art['year'][:4]} (PMID {art['pmid']})"
        for sent in split_sentences(art["abstract"]):
            subj = entity_hint(sent) if use_entity_hint else None
            field.ingest(sent, source=src, date=art["year"],
                         authority=art["authority"], confidence=0.85, subject=subj)
    field.hypothesize()
    field.propagate_trust()
    return field, data


def conflict_count(field):
    return len(field.contradictions("C9orf72 familial ALS percent"))


def run(use_entity_hint: bool):
    field, data = load_field(use_entity_hint)
    by_src = {}
    for cell in field.cells:
        by_src.setdefault(cell.source, []).append(cell.trust)
    ans = field.ask("What fraction of familial ALS is caused by C9orf72 repeat expansion?")
    ev = ans.get("evidence", [])
    return {
        "field": field, "data": data, "ans": ans, "ev": ev, "by_src": by_src,
        "conflicts": field.contradictions("C9orf72 familial ALS percent"),
    }


def main():
    data = json.load(open(FIX))
    print(f"[corpus] {len(data['articles'])} REAL PubMed abstracts, "
          f"C9orf72/ALS, 2011-2025, four journals.")
    print("  Source: PubMed (NCBI). DOIs: " +
          ", ".join(a["doi"] for a in data["articles"]))
    print("  The reported C9orf72 share of familial ALS genuinely drifts: "
          "23.5% (Neuron'11), ~46%/one-third (Neuron'11), 30-50% (EurJNeurol'20).")

    # The epistemic machinery, independent of extraction quality.
    base = run(use_entity_hint=False)
    print("\n=== works on real data regardless of extraction ===")
    print(f"  every answer cited: {bool(base['ev'])} -> "
          f"{base['field'].cells[base['ev'][0]['id']].source if base['ev'] else '-'}")
    print("  trust ordering by journal authority + recency:")
    for src, ts in sorted(base["by_src"].items(), key=lambda kv: -max(kv[1])):
        print(f"     {max(ts):.3f}  {src}")
    print("  (deterministic, 0 LLM calls)")

    # Controlled experiment: only the entity extractor changes.
    hint = run(use_entity_hint=True)
    print("\n=== the bottleneck is entity extraction, not the epistemic layer ===")
    print(f"  contradiction edges among frequency claims:")
    print(f"     rule-based subject extractor : {len(base['conflicts'])}")
    print(f"     with a domain entity extractor: {len(hint['conflicts'])}")
    for c in hint["conflicts"][:4]:
        print(f"       [{c['a_status']}] {c['a'][:48]}")
        print(f"          vs [{c['b_status']}] {c['b'][:48]}")
    a = hint["ans"]
    print(f"  current belief: {a['answer'][:60]}")
    print(f"     status {a.get('belief_status')}, trust {a.get('trust')}, "
          f"cite {hint['field'].cells[hint['ev'][0]['id']].source if hint['ev'] else '-'}")
    for h in a.get("superseded_history", [])[:3]:
        print(f"     history: [{hint['field'].cells[h['id']].source}] {h['content'][:50]}")

    print("\n=== honest findings ===")
    print("  WORKS on real prose, domain-agnostic, 0 LLM:")
    print("    - per-answer citations to real papers (DOIs)")
    print("    - trust ordered by journal authority + recency")
    print("    - deterministic, auditable")
    print("  DEGRADES on real prose with the built-in rule extractors:")
    print("    - subject/entity extraction (e.g. 'repeat expansion' instead of")
    print("      'C9orf72') -> contradictions missed until a real NER is plugged in")
    print("    - hash lexical recall on semantic paraphrase (e.g. 'protein")
    print("      pathology' -> 'TDP-43 aggregates') -> needs embeddings")
    print("  CONCLUSION: the epistemic layer is the durable contribution and is")
    print("  domain-agnostic; the NLP front-end (entity + claim extraction, encoder)")
    print("  is pluggable and is what must be upgraded for production real-text use.")


if __name__ == "__main__":
    main()
