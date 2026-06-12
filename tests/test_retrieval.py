"""Hybrid lexical+geometric retrieval: BM25, stemming, abstention margin."""
import numpy as np

from geocell import GeoCellField
from geocell.text import stem


def test_stem_unifies_morphology():
    assert stem("owns") == stem("owned") == stem("owner") == "own"
    assert stem("relies") == stem("relied")
    assert stem("budget") == "budget"          # short/irregular left alone


def test_bm25_breaks_ties_on_rare_term():
    """Every cell shares the common subject; only the rare discriminating
    term should decide. Pure geometry tied these; BM25 must not."""
    f = GeoCellField()
    f.ingest("Project Orion is currently on track.", source="status")
    f.ingest("Project Orion is owned by Dana Reyes.", source="charter")
    f.ingest("Project Orion belongs to the Platform department.", source="org")
    hits = f.recall("Who owns Project Orion?", k=3)
    assert hits[0]["source"] == "charter"       # the 'owned' cell wins


def test_bm25_surfaces_inferred_multihop_answer():
    f = GeoCellField()
    f.ingest("Project Orion belongs to the Platform department.", source="charter")
    f.ingest("The Platform department relies on vendor Stark.", source="vendorplan")
    f.ingest("Project Orion is currently on track.", source="status")
    f.hypothesize()
    hits = f.recall("Which vendor does Project Orion rely on?", k=3)
    assert "Stark" in hits[0]["content"]        # inferred depends_on cell ranks top


def test_recall_margin_distinguishes_confident_from_tie():
    f = GeoCellField()
    f.ingest("Project Orion is owned by Dana Reyes.", source="charter")
    f.ingest("Project Orion is currently on track.", source="status")
    f.ingest("Project Orion belongs to the Platform department.", source="org")
    confident = f.recall("Who owns Project Orion?", k=2)
    margin_conf = confident[0]["resonance"] - confident[1]["resonance"]
    # a query with no discriminating term beyond the shared subject
    tie = f.recall("Tell me something about Project Orion.", k=2)
    margin_tie = tie[0]["resonance"] - tie[1]["resonance"]
    assert margin_conf > margin_tie


def test_index_rebuilt_after_compact_roundtrip(tmp_path):
    f = GeoCellField()
    f.ingest("Project Orion is owned by Dana Reyes.", source="charter")
    f.ingest("Project Orion is currently on track.", source="status")
    path = str(tmp_path / "f.geocell")
    f.save_compact(path)
    g = GeoCellField.load_compact(path)
    assert g._bm25("owned").sum() > 0           # postings rebuilt on load
    assert g.recall("Who owns Project Orion?", k=1)[0]["source"] == "charter"
