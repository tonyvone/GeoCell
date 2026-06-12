"""GeoCellField: the epistemic particle field.

The engine treats memory as physics:

- Ingestion drops a new cell into the field and wires trinary relations
  (support / unknown / contradiction) to its neighborhood.
- Supersession resolves dated contradictions: a newer claim carrying a
  revision signal (or higher authority) displaces the older belief,
  which is kept as history rather than deleted.
- propagate_trust() runs a damped fixed-point iteration: trust flows
  along support edges and unresolved contradictions exert pressure.
- settle() relaxes the geometry: corroborating cells attract,
  contradicting cells repel, so position comes to encode evidential
  structure rather than just wording.
- recall() seeds activation geometrically and lets it diffuse across
  support edges, giving multi-hop retrieval.
- hypothesize() applies defeasible transitive rules and writes the
  conclusions back into the field as inferred cells with provenance.
- consolidate() merges tight clusters of mutually supporting memories
  into concept cells.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

from geocell.cell import (
    ACTIVE,
    CONCEPT,
    CONTESTED,
    CONTRADICTION,
    GeoCell,
    INFERRED,
    OBSERVED,
    RETRACTED,
    SUPERSEDED,
    SUPPORT,
    UNKNOWN,
)
from geocell.compress import (
    asymmetric_similarity,
    decode_b64,
    encode_b64,
    hamming_similarity,
    index_footprint,
    pack_signs,
    unpack_signs,
)
from geocell.encoding import cosine, encode
from geocell.text import (
    HISTORICAL_WORDS,
    NEGATORS,
    NUMERIC_QUERY_WORDS,
    REVISION_WORDS,
    extract_claim_values,
    extract_relations,
    extract_subject,
    raw_words,
    stem,
    tokenize,
)

_STATUS_WEIGHT = {ACTIVE: 1.0, CONTESTED: 0.8, SUPERSEDED: 0.35, RETRACTED: 0.1}
_STATUS_WEIGHT_HISTORICAL = {ACTIVE: 0.9, CONTESTED: 0.85, SUPERSEDED: 1.15, RETRACTED: 0.2}


class GeoCellField:
    def __init__(self, dims: int = 768, support_threshold: float = 0.10):
        self.dims = dims
        self.support_threshold = support_threshold
        self.cells: List[GeoCell] = []
        self.graph = nx.Graph()
        self._trust_trace: Dict[int, Dict[str, float]] = {}
        self._trust_dirty = True
        self._anchors: Dict[int, np.ndarray] = {}
        self.quantized = False
        self._bits_pos = np.zeros((0, dims // 8), dtype=np.uint8)
        self._bits_anchor = np.zeros((0, dims // 8), dtype=np.uint8)
        # Inverted index for the BM25 lexical channel. The geometry says
        # how similar two sentences are overall; BM25 says which rare,
        # informative words they share — so a discriminating term ("vendor",
        # "owned") outweighs the common subject every candidate repeats.
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._doc_len: Dict[int, int] = {}
        self._total_len = 0
        # True when loaded from a compact snapshot: float positions are
        # rotated-space placeholders, so all similarity must use bits and
        # settle must first re-derive geometry from the lexical anchors.
        self._serving_only = False

    def _anchor(self, c: GeoCell) -> np.ndarray:
        """The cell's lexical anchor: the deterministic encoding of its
        content, unaffected by field relaxation."""
        if c.id not in self._anchors:
            self._anchors[c.id] = encode(c.content, self.dims)
        return self._anchors[c.id]

    # ------------------------------------------------------------------
    # Ingestion and relation wiring
    # ------------------------------------------------------------------

    def ingest(self, content: str, source: str = "user", date: str = "",
               confidence: float = 0.75, authority: float = 0.5,
               kind: str = OBSERVED, parents: Optional[List[int]] = None) -> int:
        content = content.strip()
        pos = encode(content, self.dims)
        toks = set(tokenize(content))
        cell = GeoCell(
            id=len(self.cells),
            content=content,
            position=pos.tolist(),
            radius=0.20,
            created_at=time.time(),
            source=source,
            date=date,
            confidence=max(0.0, min(float(confidence), 1.0)),
            authority=max(0.0, min(float(authority), 1.0)),
            subject=extract_subject(content),
            values=extract_claim_values(content),
            polarity=-1 if toks & NEGATORS else 1,
            revision_signal=bool(toks & REVISION_WORDS),
            kind=kind,
            parents=list(parents or []),
        )
        cell.log(f"created kind={kind} source={source} date={date or '?'}")
        self.cells.append(cell)
        self._index_cell(cell)
        self.graph.add_node(cell.id)
        for pid in cell.parents:
            self.graph.add_edge(cell.id, pid, state=SUPPORT, weight=0.9,
                                reason="derivation", resolved=False)
        if self.quantized:
            self._bits_pos = np.vstack([self._bits_pos, pack_signs(pos)])
            self._bits_anchor = np.vstack([self._bits_anchor, pack_signs(self._anchor(cell))])
        self._wire(cell)
        self._trust_dirty = True
        return cell.id

    def _index_cell(self, cell: GeoCell) -> None:
        toks = [stem(t) for t in tokenize(cell.content)]
        self._doc_len[cell.id] = len(toks)
        self._total_len += len(toks)
        for tok, tf in Counter(toks).items():
            self._postings[tok][cell.id] = tf

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Okapi BM25 over the inverted index. Standard IR ranking with
        tf saturation and length normalization, computed against the same
        stemmed vocabulary the encoder uses."""
        n = len(self.cells)
        scores = np.zeros(n, dtype=np.float64)
        if not self._doc_len:
            return scores
        avgdl = self._total_len / max(1, len(self._doc_len))
        for tok in {stem(t) for t in tokenize(query)}:
            postings = self._postings.get(tok)
            if not postings:
                continue
            df = len(postings)
            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for cid, tf in postings.items():
                dl = self._doc_len.get(cid, 0)
                scores[cid] += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
        return scores

    @staticmethod
    def _subject_overlap(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(1, min(len(sa), len(sb)))

    @staticmethod
    def _value_conflict(a: GeoCell, b: GeoCell) -> bool:
        if not a.values or not b.values:
            return False
        av = {round(float(x["value"] or 0), 4) for x in a.values}
        bv = {round(float(x["value"] or 0), 4) for x in b.values}
        return not (av & bv)

    @staticmethod
    def _content_overlap(a: GeoCell, b: GeoCell) -> float:
        ta = set(tokenize(a.content)) - NEGATORS
        tb = set(tokenize(b.content)) - NEGATORS
        return len(ta & tb) / max(1, min(len(ta), len(tb)))

    def _pair_sim(self, a: GeoCell, b: GeoCell) -> float:
        if self.quantized and max(a.id, b.id) < len(self._bits_pos):
            return float(hamming_similarity(self._bits_pos[a.id],
                                            self._bits_pos[[b.id]], self.dims)[0])
        return cosine(np.array(a.position, dtype=np.float32), np.array(b.position, dtype=np.float32))

    def _relation(self, a: GeoCell, b: GeoCell) -> Tuple[int, float, str]:
        sim = self._pair_sim(a, b)
        same_subject = self._subject_overlap(a.subject, b.subject) >= 0.99
        # Differing numbers only conflict when the sentences describe the
        # same quantity: "Orion budget is $2M" vs "Orion headcount is 50"
        # share a subject but are not in dispute.
        if same_subject and self._value_conflict(a, b) and self._content_overlap(a, b) >= 0.55:
            return CONTRADICTION, max(0.15, 1.0 - sim), "claim_conflict"
        # Opposite polarity only contradicts when both sentences talk
        # about the same predicate, not merely the same subject.
        if same_subject and a.polarity != b.polarity and self._content_overlap(a, b) >= 0.7:
            return CONTRADICTION, max(0.15, 1.0 - sim), "polarity_conflict"
        if sim >= self.support_threshold or same_subject:
            boost = 0.20 if same_subject else 0.0
            return SUPPORT, max(0.01, sim + boost), "support"
        return UNKNOWN, 0.0, "unknown"

    def _wire(self, new: GeoCell) -> None:
        for other in self.cells[:-1]:
            if other.id in new.parents:
                continue
            state, weight, reason = self._relation(new, other)
            if state == UNKNOWN:
                continue
            self.graph.add_edge(new.id, other.id, state=state, weight=float(weight),
                                reason=reason, resolved=False)
            if state == SUPPORT:
                new.radius = max(0.08, new.radius * 0.98)
            else:
                new.radius = min(0.45, new.radius * 1.08)
                self._adjudicate(new, other)

    # ------------------------------------------------------------------
    # Belief revision
    # ------------------------------------------------------------------

    def _adjudicate(self, a: GeoCell, b: GeoCell) -> None:
        """Resolve a contradiction if the evidence clearly favors one side.

        A dated revision (or a strictly stronger authority arriving later)
        supersedes the older claim. Otherwise both cells become contested
        until something stronger arrives.
        """
        # A conflict with an already-displaced belief is settled history,
        # not an open dispute.
        if a.status in (SUPERSEDED, RETRACTED) or b.status in (SUPERSEDED, RETRACTED):
            if self.graph.has_edge(a.id, b.id):
                self.graph.edges[a.id, b.id]["resolved"] = True
            return
        # A hypothesis never beats an observation: defeasible reasoning.
        if a.kind == INFERRED and b.kind == OBSERVED:
            self._retract(a, b)
            return
        if b.kind == INFERRED and a.kind == OBSERVED:
            self._retract(b, a)
            return

        newer, older = (a, b) if (a.date or "") >= (b.date or "") else (b, a)
        decisive = newer.date and older.date and newer.date != older.date and (
            newer.revision_signal or newer.authority > older.authority
        )
        # A later but clearly weaker claim (a rumor against a board update)
        # is rejected outright instead of contesting the stronger belief.
        if not decisive and older.authority >= newer.authority + 0.25:
            self._supersede(loser=newer, winner=older)
            return
        if decisive:
            self._supersede(loser=older, winner=newer)
            # The winner also displaces anything the loser had already
            # superseded-equivalent claims (same values as the loser).
            for nb in list(self.graph.neighbors(older.id)):
                other = self.cells[nb]
                if other.id == newer.id or other.status != ACTIVE:
                    continue
                d = self.graph.edges[older.id, nb]
                if d.get("state") == SUPPORT and not self._value_conflict(older, other) \
                        and self._subject_overlap(older.subject, other.subject) >= 0.99 \
                        and other.values and older.values:
                    self._supersede(loser=other, winner=newer)
                    if self.graph.has_edge(newer.id, other.id):
                        self.graph.edges[newer.id, other.id]["resolved"] = True
        else:
            for c in (a, b):
                if c.status == ACTIVE:
                    c.status = CONTESTED
                    c.log(f"contested by #{b.id if c is a else a.id}")

    def _supersede(self, loser: GeoCell, winner: GeoCell) -> None:
        if loser.status == SUPERSEDED:
            return
        loser.status = SUPERSEDED
        loser.log(f"superseded by #{winner.id} ({winner.source} {winner.date})")
        winner.log(f"supersedes #{loser.id} ({loser.source} {loser.date})")
        if self.graph.has_edge(winner.id, loser.id):
            self.graph.edges[winner.id, loser.id]["resolved"] = True
            self.graph.edges[winner.id, loser.id]["reason"] = "superseded"

    def _retract(self, hypothesis: GeoCell, evidence: GeoCell) -> None:
        hypothesis.status = RETRACTED
        hypothesis.log(f"retracted: contradicts observed #{evidence.id}")
        if self.graph.has_edge(hypothesis.id, evidence.id):
            self.graph.edges[hypothesis.id, evidence.id]["resolved"] = True

    # ------------------------------------------------------------------
    # Trust propagation
    # ------------------------------------------------------------------

    def _prior(self, c: GeoCell) -> float:
        prior = 0.55 * c.confidence + 0.45 * c.authority
        if c.kind == INFERRED and c.parents:
            parent_prior = min(self._prior(self.cells[p]) for p in c.parents)
            prior = min(prior, parent_prior) * 0.85
        return prior

    def propagate_trust(self, iterations: int = 12) -> Dict[int, float]:
        """Damped fixed-point iteration. Trust flows along support edges;
        unresolved contradictions with a more-trusted opponent push trust
        down. Superseded and retracted cells are capped."""
        n = len(self.cells)
        if not n:
            self._trust_dirty = False
            return {}
        priors = np.array([self._prior(c) for c in self.cells], dtype=np.float64)
        trust = priors.copy()
        for _ in range(iterations):
            nxt = trust.copy()
            for i, c in enumerate(self.cells):
                sup_num, sup_den, pressure = 0.0, 0.0, 0.0
                for j in self.graph.neighbors(i):
                    d = self.graph.edges[i, j]
                    w = float(d.get("weight", 0.0))
                    if d.get("state") == SUPPORT:
                        sup_num += w * trust[j]
                        sup_den += w
                    elif d.get("state") == CONTRADICTION and not d.get("resolved"):
                        pressure = max(pressure, trust[j] - trust[i])
                support = sup_num / sup_den if sup_den else priors[i]
                value = 0.55 * priors[i] + 0.45 * support - 0.30 * max(0.0, pressure)
                if c.status == SUPERSEDED:
                    value = min(value, 0.30)
                elif c.status == RETRACTED:
                    value = min(value, 0.10)
                nxt[i] = min(0.99, max(0.02, value))
                self._trust_trace[i] = {
                    "prior": round(priors[i], 4),
                    "support_flow": round(support, 4),
                    "contradiction_pressure": round(max(0.0, pressure), 4),
                }
            if np.max(np.abs(nxt - trust)) < 1e-5:
                trust = nxt
                break
            trust = 0.5 * trust + 0.5 * nxt
        for i, c in enumerate(self.cells):
            c.trust = float(trust[i]) if n else 0.5
        self._trust_dirty = False
        return {c.id: round(c.trust, 4) for c in self.cells}

    def _ensure_trust(self) -> None:
        if self._trust_dirty and self.cells:
            self.propagate_trust()

    # ------------------------------------------------------------------
    # Field relaxation: geometry shaped by evidence
    # ------------------------------------------------------------------

    def settle(self, steps: int = 30, lr: float = 0.05) -> Dict[str, Any]:
        """Relax the field. Support edges pull cells together with force
        proportional to edge weight and mutual trust; unresolved
        contradictions push apart. Positions stay on the unit sphere.

        Every cell is also tethered to its lexical anchor (the deterministic
        encoding of its content) by a spring, so evidence deforms the
        geometry without ever detaching a memory from what it says.

        After settling, geometric distance encodes corroboration: a tight
        cluster is a body of mutually reinforcing evidence, and the gap
        between clusters is epistemic, not just lexical.
        """
        self._ensure_trust()
        n = len(self.cells)
        if n < 2:
            return {"steps": 0, "displacement": 0.0}
        anchors = np.array([self._anchor(c) for c in self.cells], dtype=np.float64)
        if self._serving_only:
            # Snapshot positions are serving placeholders; re-derive real
            # geometry from the lexical anchors before relaxing.
            P = anchors.copy()
            self._serving_only = False
        else:
            P = np.array([c.position for c in self.cells], dtype=np.float64)
        degree = np.array([max(1, self.graph.degree(i)) for i in range(n)], dtype=np.float64)
        start = P.copy()
        for _ in range(steps):
            delta = (2.0 * lr) * (anchors - P)
            for u, v, d in self.graph.edges(data=True):
                diff = P[v] - P[u]
                w = float(d.get("weight", 0.0))
                if d.get("state") == SUPPORT:
                    f = lr * w * 0.5 * (self.cells[u].trust + self.cells[v].trust)
                    delta[u] += (f / degree[u]) * diff
                    delta[v] -= (f / degree[v]) * diff
                elif d.get("state") == CONTRADICTION and not d.get("resolved"):
                    f = lr * 0.5 * w
                    delta[u] -= (f / degree[u]) * diff
                    delta[v] += (f / degree[v]) * diff
            P += delta
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P /= norms
        displacement = float(np.linalg.norm(P - start, axis=1).mean())
        for i, c in enumerate(self.cells):
            c.position = P[i].tolist()
            if self.quantized:
                self._bits_pos[i] = pack_signs(P[i])
            sup = sum(1 for j in self.graph.neighbors(i) if self.graph.edges[i, j].get("state") == SUPPORT)
            con = sum(1 for j in self.graph.neighbors(i)
                      if self.graph.edges[i, j].get("state") == CONTRADICTION and not self.graph.edges[i, j].get("resolved"))
            c.radius = float(np.clip(0.20 - 0.015 * sup + 0.04 * con, 0.06, 0.50))
        return {"steps": steps, "displacement": round(displacement, 6)}

    # ------------------------------------------------------------------
    # Recall: geometric seeding + spreading activation
    # ------------------------------------------------------------------

    def _support_matrix(self) -> np.ndarray:
        n = len(self.cells)
        A = np.zeros((n, n), dtype=np.float64)
        for u, v, d in self.graph.edges(data=True):
            if d.get("state") == SUPPORT:
                A[u, v] = A[v, u] = float(d.get("weight", 0.0))
        row = A.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return A / row

    def recall(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        if not self.cells:
            return []
        self._ensure_trust()
        q = encode(query, self.dims)
        qtoks = set(tokenize(query))
        historical = bool(set(raw_words(query)) & HISTORICAL_WORDS)
        status_weight = _STATUS_WEIGHT_HISTORICAL if historical else _STATUS_WEIGHT

        # Resonance blends the lexical anchor (what the memory says) with
        # the settled position (where the evidence moved it). In quantized
        # mode both are 96-byte sign signatures compared via XOR+popcount.
        if self.quantized:
            sims = 0.65 * asymmetric_similarity(q, self._bits_anchor, self.dims) + \
                   0.35 * asymmetric_similarity(q, self._bits_pos, self.dims)
        # Lexical channel: BM25, normalized so it blends with the geometric
        # cosine. It carries the term-informativeness the hash geometry
        # lacks, which is what breaks ties between sibling sentences.
        bm = self._bm25(query)
        bmax = bm.max()
        if bmax > 0:
            bm = bm / bmax

        base = np.zeros(len(self.cells), dtype=np.float64)
        for c in self.cells:
            if self.quantized:
                sim = float(sims[c.id])
            else:
                sim = 0.65 * cosine(q, self._anchor(c)) + \
                      0.35 * cosine(q, np.array(c.position, dtype=np.float32))
            subject_terms = set(c.subject.split())
            if subject_terms and subject_terms <= qtoks:
                bonus = 0.16
            elif qtoks & subject_terms:
                bonus = 0.04
            else:
                bonus = 0.0
            base[c.id] = max(0.0, 0.55 * sim + 0.45 * bm[c.id] + bonus)

        # Spreading activation: energy diffuses two hops along support edges,
        # so a query can light up memories it never mentions.
        A = self._support_matrix()
        act = base + 0.30 * (A @ base) + 0.12 * (A @ (A @ base))

        scored = []
        for c in self.cells:
            # A historical question asks what *was* believed, so the
            # current trust level must not bury displaced beliefs.
            trust_gate = 1.0 if historical else 0.45 + 0.55 * c.trust
            score = act[c.id] * trust_gate * status_weight.get(c.status, 0.5)
            scored.append((score, c))
        scored.sort(key=lambda x: (x[0], x[1].id), reverse=True)
        return [c.view(score) for score, c in scored[:k]]

    # ------------------------------------------------------------------
    # Asking: answer + belief state + reasoning trace
    # ------------------------------------------------------------------

    def contradictions(self, query: str = "", include_resolved: bool = True) -> List[Dict[str, Any]]:
        ids = set(range(len(self.cells))) if not query else {r["id"] for r in self.recall(query, k=20)}
        out = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("state") != CONTRADICTION or (u not in ids and v not in ids):
                continue
            if not include_resolved and d.get("resolved"):
                continue
            out.append({
                "a_id": u, "a": self.cells[u].content, "a_status": self.cells[u].status,
                "a_date": self.cells[u].date, "a_source": self.cells[u].source,
                "b_id": v, "b": self.cells[v].content, "b_status": self.cells[v].status,
                "b_date": self.cells[v].date, "b_source": self.cells[v].source,
                "reason": d.get("reason"), "resolved": bool(d.get("resolved")),
                "weight": round(float(d.get("weight", 0)), 4),
            })
        return out

    def ask(self, query: str) -> Dict[str, Any]:
        hits = self.recall(query, k=10)
        if not hits:
            return {"answer": "No relevant memory found.", "confidence": 0.0, "evidence": []}
        qtoks = set(tokenize(query))
        # Concept cells are abstractions for navigation, not answers:
        # only fall back to one when nothing concrete resonates.
        concrete = [h for h in hits if self.cells[h["id"]].kind != CONCEPT]
        candidates = concrete or hits
        hits = candidates
        if qtoks & NUMERIC_QUERY_WORDS:
            # Prefer memories carrying numeric claims, but never let a
            # weakly relevant number outrank the actual best match.
            valued = [h for h in hits if h["values"] and h["resonance"] >= 0.5 * hits[0]["resonance"]]
            if valued:
                candidates = valued
        else:
            # Relation-aware selection: when the query names a predicate,
            # prefer memories that actually express that relation.
            qraw = set(raw_words(query))
            rel_query = None
            if qraw & {"depends", "depend"}:
                rel_query = "depends_on"
            elif "part" in qraw:
                rel_query = "part_of"
            elif qraw & {"requires", "require"}:
                rel_query = "requires"
            if rel_query:
                rel_hits = [h for h in candidates
                            if any(t["relation"] == rel_query
                                   for t in extract_relations(self.cells[h["id"]].content))]
                if rel_hits:
                    candidates = rel_hits
        # Resonance ranked at 2-decimal granularity: among near-ties the
        # belief layer (trust, recency) decides, not encoding jitter.
        best = max(candidates, key=lambda h: (round(h["resonance"], 2), h["trust"], h["date"] or "0000-00-00"))
        cell = self.cells[best["id"]]

        confidence = min(0.99, max(0.05, best["resonance"] * 0.45 + cell.trust * 0.55))
        answer: Dict[str, Any] = {
            "answer": cell.content,
            "belief_status": cell.status,
            "trust": round(cell.trust, 4),
            "confidence": round(confidence, 3),
            "method": "spreading_activation + trust_propagation + belief_revision",
            "evidence": hits[:5],
        }
        if cell.kind == INFERRED:
            answer["reasoning_chain"] = self._derivation_chain(cell)
        superseded = [self.cells[h["id"]].view(h["resonance"]) for h in hits
                      if self.cells[h["id"]].status == SUPERSEDED
                      and self._subject_overlap(self.cells[h["id"]].subject, cell.subject) >= 0.99
                      and h["id"] != cell.id]
        if superseded:
            answer["superseded_history"] = superseded[:3]
        conflicts = self.contradictions(query, include_resolved=False)
        if conflicts:
            answer["open_contradictions"] = conflicts[:5]
        return answer

    def _derivation_chain(self, cell: GeoCell) -> List[Dict[str, Any]]:
        chain, seen = [], set()
        frontier = [cell.id]
        while frontier:
            cid = frontier.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            c = self.cells[cid]
            chain.append({"id": c.id, "content": c.content, "kind": c.kind, "source": c.source})
            frontier.extend(c.parents)
        return chain

    # ------------------------------------------------------------------
    # Defeasible inference: hypotheses written back into the field
    # ------------------------------------------------------------------

    _RULES = {
        ("part_of", "depends_on"): "depends_on",
        ("part_of", "part_of"): "part_of",
        ("depends_on", "depends_on"): "depends_on",
        ("part_of", "requires"): "requires",
    }
    _RELATION_SURFACE = {"part_of": "is part of", "depends_on": "depends on", "requires": "requires"}

    def hypothesize(self, max_rounds: int = 3) -> List[Dict[str, Any]]:
        """Transitive closure over extracted relations. Each conclusion
        becomes an inferred cell with provenance edges to its parents.
        Hypotheses that contradict observed evidence are retracted on
        arrival (handled by _adjudicate)."""
        created: List[Dict[str, Any]] = []
        for _ in range(max_rounds):
            triples = self._collect_triples()
            existing: Set[Tuple[str, str, str]] = {(t["relation"], t["head_key"], t["tail_key"]) for t, _ in triples}
            new_round = 0
            for t1, id1 in triples:
                for t2, id2 in triples:
                    rule = self._RULES.get((t1["relation"], t2["relation"]))
                    if not rule or t1["tail_key"] != t2["head_key"]:
                        continue
                    key = (rule, t1["head_key"], t2["tail_key"])
                    if key in existing or t1["head_key"] == t2["tail_key"]:
                        continue
                    p1, p2 = self.cells[id1], self.cells[id2]
                    content = f"{t1['head']} {self._RELATION_SURFACE[rule]} {t2['tail']}."
                    cid = self.ingest(
                        content,
                        source="geocell_inference",
                        date=max(p1.date, p2.date),
                        confidence=min(p1.confidence, p2.confidence) * 0.9,
                        authority=min(p1.authority, p2.authority) * 0.9,
                        kind=INFERRED,
                        parents=[id1, id2],
                    )
                    existing.add(key)
                    new_round += 1
                    created.append(self.cells[cid].view())
            if not new_round:
                break
        return created

    def _collect_triples(self) -> List[Tuple[Dict[str, str], int]]:
        out = []
        for c in self.cells:
            if c.status in (SUPERSEDED, RETRACTED):
                continue
            for t in extract_relations(c.content):
                out.append((t, c.id))
        return out

    # ------------------------------------------------------------------
    # Consolidation: clusters become concepts
    # ------------------------------------------------------------------

    def consolidate(self, min_size: int = 3) -> List[Dict[str, Any]]:
        """Find communities of mutually supporting observed cells and
        summarize each as a concept cell at the cluster centroid."""
        self._ensure_trust()
        sub = nx.Graph()
        for c in self.cells:
            if c.kind == OBSERVED and c.status in (ACTIVE, CONTESTED):
                sub.add_node(c.id)
        for u, v, d in self.graph.edges(data=True):
            if d.get("state") == SUPPORT and u in sub and v in sub:
                sub.add_edge(u, v, weight=float(d.get("weight", 0.0)))
        if sub.number_of_edges() == 0:
            return []
        created = []
        communities = nx.algorithms.community.greedy_modularity_communities(sub, weight="weight")
        for com in communities:
            members = sorted(com)
            if len(members) < min_size:
                continue
            if any(set(members) <= set(c.parents) for c in self.cells if c.kind == CONCEPT):
                continue
            counts: Dict[str, int] = {}
            for m in members:
                for t in set(tokenize(self.cells[m].content)):
                    counts[t] = counts.get(t, 0) + 1
            theme = [t for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n >= 2][:5]
            content = f"Concept: {' '.join(theme) if theme else 'cluster'}."
            cid = self.ingest(
                content,
                source="geocell_consolidation",
                confidence=float(np.mean([self.cells[m].confidence for m in members])),
                authority=float(np.mean([self.cells[m].authority for m in members])),
                kind=CONCEPT,
                parents=members,
            )
            centroid = np.mean([np.array(self.cells[m].position) for m in members], axis=0)
            norm = float(np.linalg.norm(centroid))
            self.cells[cid].position = (centroid / norm).tolist() if norm else self.cells[cid].position
            created.append(self.cells[cid].view())
        return created

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def path(self, start: str, end: str) -> Optional[List[Dict[str, Any]]]:
        s, e = self.recall(start, 1), self.recall(end, 1)
        if not s or not e:
            return None
        self._ensure_trust()

        def cost(u: int, v: int, d: Dict[str, Any]) -> float:
            if d.get("state") == CONTRADICTION:
                return 25.0
            w = max(0.01, float(d.get("weight", 0.01)))
            t = max(0.05, 0.5 * (self.cells[u].trust + self.cells[v].trust))
            return 1.0 / (w * t)

        try:
            route = nx.shortest_path(self.graph, s[0]["id"], e[0]["id"], weight=cost)
            return [self.cells[i].view() for i in route]
        except nx.NetworkXNoPath:
            return None

    def inspect(self, cell_id: int) -> Dict[str, Any]:
        c = self.cells[cell_id]
        neighbors = []
        for n in self.graph.neighbors(cell_id):
            d = self.graph.edges[cell_id, n]
            neighbors.append({
                "id": n, "content": self.cells[n].content, "state": d.get("state"),
                "weight": round(float(d.get("weight", 0)), 4),
                "reason": d.get("reason"), "resolved": bool(d.get("resolved")),
            })
        return {"cell": c.view(), "neighbors": sorted(neighbors, key=lambda x: x["weight"], reverse=True)}

    def why(self, cell_id: int) -> Dict[str, Any]:
        """Full epistemic accounting for one belief: where it came from,
        what happened to it, what holds it up, and what pushes against it."""
        self._ensure_trust()
        c = self.cells[cell_id]
        out = self.inspect(cell_id)
        out["ledger"] = list(c.ledger)
        out["trust_breakdown"] = self._trust_trace.get(cell_id, {})
        if c.parents:
            out["derived_from"] = [self.cells[p].view() for p in c.parents]
        return out

    def timeline(self, subject: str) -> List[Dict[str, Any]]:
        """Evolution of belief about a subject, oldest first."""
        key = " ".join(tokenize(subject))
        rows = [c for c in self.cells if self._subject_overlap(c.subject, key) >= 0.5]
        rows.sort(key=lambda c: (c.date or "9999-99-99", c.id))
        return [c.view() for c in rows]

    def stats(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for c in self.cells:
            by_status[c.status] = by_status.get(c.status, 0) + 1
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        open_conflicts = sum(1 for _, _, d in self.graph.edges(data=True)
                             if d.get("state") == CONTRADICTION and not d.get("resolved"))
        return {
            "cells": len(self.cells),
            "edges": self.graph.number_of_edges(),
            "by_status": by_status,
            "by_kind": by_kind,
            "open_contradictions": open_conflicts,
        }

    # ------------------------------------------------------------------
    # Quantization: the 96-bytes-per-memory serving form
    # ------------------------------------------------------------------

    def quantize(self) -> Dict[str, Any]:
        """Freeze the field for serving: every position and anchor is
        snapped to its sign pattern and recall switches to Hamming
        similarity over packed bits. Lossless for the belief layer
        (graph, statuses, trust); near-lossless for ranking."""
        n = len(self.cells)
        self._bits_pos = np.zeros((n, self.dims // 8), dtype=np.uint8)
        self._bits_anchor = np.zeros((n, self.dims // 8), dtype=np.uint8)
        for c in self.cells:
            self._bits_pos[c.id] = pack_signs(np.array(c.position, dtype=np.float32))
            self._bits_anchor[c.id] = pack_signs(self._anchor(c))
        self.quantized = True
        float_b, bin_b, ratio = index_footprint(n, self.dims)
        return {"cells": n, "index_bytes_float32": float_b,
                "index_bytes_binary": bin_b, "compression": round(ratio, 1)}

    def _skeleton_edges(self, max_support_degree: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """The epistemic skeleton: every contradiction and derivation edge,
        plus each node's strongest support edges. The dense support lattice
        is redundant for serving — spreading activation only needs the
        strong links."""
        keep: Set[Tuple[int, int]] = set()
        per_node: Dict[int, List[Tuple[float, int, int]]] = {}
        out = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("state") != SUPPORT or d.get("reason") == "derivation":
                keep.add((u, v))
                continue
            w = float(d.get("weight", 0.0))
            per_node.setdefault(u, []).append((w, u, v))
            per_node.setdefault(v, []).append((w, u, v))
        for ranked in per_node.values():
            ranked.sort(reverse=True)
            for w, u, v in ranked[:max_support_degree]:
                keep.add((u, v))
        for u, v, d in self.graph.edges(data=True):
            if (u, v) in keep:
                out.append((u, v, d))
        return out

    def save_compact(self, path: str, max_support_degree: int = 16) -> Dict[str, Any]:
        """Binary snapshot: cells keep text + provenance + lifecycle, but
        geometry is stored as base64 sign signatures and the support
        lattice is pruned to its skeleton. Loads back as a quantized
        (serving) field. Non-destructive: the in-memory field keeps its
        full float geometry and full graph."""
        if not self.quantized:
            self.quantize()
        cells = []
        for c in self.cells:
            d = asdict(c)
            d.pop("position")
            d["bits_pos"] = encode_b64(self._bits_pos[c.id])
            d["bits_anchor"] = encode_b64(self._bits_anchor[c.id])
            cells.append(d)
        edges = self._skeleton_edges(max_support_degree)
        reasons = sorted({d.get("reason", "") for _, _, d in edges})
        reason_id = {r: i for i, r in enumerate(reasons)}
        data = {
            "version": 4, "format": "compact", "dims": self.dims,
            "support_threshold": self.support_threshold,
            "cells": cells,
            "edge_reasons": reasons,
            "edges_packed": [[u, v, d.get("state", 0), round(float(d.get("weight", 0.0)), 3),
                              reason_id[d.get("reason", "")], int(bool(d.get("resolved")))]
                             for u, v, d in edges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        return {"path": path, "bytes": os.path.getsize(path),
                "edges_kept": len(edges), "edges_total": self.graph.number_of_edges()}

    @classmethod
    def load_compact(cls, path: str) -> "GeoCellField":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mem = cls(dims=data["dims"], support_threshold=data.get("support_threshold", 0.10))
        n = len(data["cells"])
        mem._bits_pos = np.zeros((n, mem.dims // 8), dtype=np.uint8)
        mem._bits_anchor = np.zeros((n, mem.dims // 8), dtype=np.uint8)
        for raw in data["cells"]:
            bits_pos = decode_b64(raw.pop("bits_pos"))
            bits_anchor = decode_b64(raw.pop("bits_anchor"))
            raw["position"] = unpack_signs(bits_pos, mem.dims).tolist()
            cell = GeoCell(**raw)
            mem.cells.append(cell)
            mem._index_cell(cell)
            mem.graph.add_node(cell.id)
            mem._bits_pos[cell.id] = bits_pos
            mem._bits_anchor[cell.id] = bits_anchor
            mem._anchors[cell.id] = unpack_signs(bits_anchor, mem.dims)
        reasons = data.get("edge_reasons", [])
        for u, v, state, weight, rid, resolved in data.get("edges_packed", []):
            mem.graph.add_edge(u, v, state=state, weight=weight,
                               reason=reasons[rid], resolved=bool(resolved))
        mem.quantized = True
        mem._serving_only = True
        mem._trust_dirty = True
        return mem

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "dims": self.dims,
            "support_threshold": self.support_threshold,
            "cells": [asdict(c) for c in self.cells],
            "edges": [(u, v, d) for u, v, d in self.graph.edges(data=True)],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoCellField":
        mem = cls(dims=data.get("dims", 768), support_threshold=data.get("support_threshold", 0.10))
        defaults = {"kind": OBSERVED, "status": ACTIVE, "trust": 0.5, "parents": [], "ledger": []}
        for raw in data.get("cells", []):
            mem.cells.append(GeoCell(**{**defaults, **raw}))
        for c in mem.cells:
            mem._index_cell(c)
            mem.graph.add_node(c.id)
        for u, v, d in data.get("edges", []):
            mem.graph.add_edge(u, v, **d)
        mem._trust_dirty = True
        return mem

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GeoCellField":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
