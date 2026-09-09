"""
Signal reasoning: turn sourced, stance-tagged claims about ONE asset into an
explained suggestion — `suggest_buy`, `suggest_avoid`, or `no_call`.

This is the reasoning layer for a research/advisory agent, NOT an execution
layer. It never sizes, orders, or shorts. It answers one question:

    Given everything the sources say about this asset, and the way those
    claims support and contradict each other, which thesis is best backed
    right now — and is the backing strong and diverse enough to act on?

How it works (reuses the contradiction engine unchanged):

  1. Every claim becomes a node seeded with its source's reliability. Claims
     carry a `stance`: "bullish" or "bearish" for THESIS claims (a position on
     the asset), "neutral" for EVIDENCE claims (a fact that supports or refutes
     a thesis but is not itself a position).
  2. Verified support/contradiction relationships become typed edges.
  3. The engine propagates confidence (support lifts, contradiction penalizes),
     detects the contradictions, and re-ranks the contested pairs.
  4. Each side's score is its best-backed thesis claim's FINAL confidence. The
     margin between the bull and bear scores decides the call.
  5. Two abstention guards keep the layer honest:
       - thin margin  -> no_call  (the evidence does not separate the theses)
       - single source -> no_call (a winning thesis backed by only one distinct
         source is not corroborated; this is the collusion/echo guard from the
         README's limitations, in its simplest form)

Everything the call rests on is returned in the Suggestion so a human (or a
downstream agent) can audit it: both theses, their scores, the sources behind
each, the contradictions found, and the full per-claim ranking.

Evaluation discipline: `log_suggestion` appends each call with its `as_of`
date to a JSONL decision log, so forward hit-rate can be measured later against
realized returns net of costs. A suggestion that is never scored forward is
just an opinion.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .graph import ReasoningGraph, RelationType

STANCES = ("bullish", "bearish", "neutral")
THESIS_STANCES = ("bullish", "bearish")

_RELATION_MAP = {
    "support": RelationType.SUPPORT,
    "contradiction": RelationType.CONTRADICTION,
}

_SUPPORT_TYPES = {RelationType.SUPPORT.value, RelationType.CONFLUENCE.value,
                  RelationType.VALID_INFERENCE.value, RelationType.DERIVATION.value}


@dataclass
class Suggestion:
    """The engine's explained call on one asset."""
    asset: str
    action: str                    # suggest_buy | suggest_avoid | no_call
    bull_score: float
    bear_score: float
    margin: float                  # bull_score - bear_score
    bull_thesis: str | None        # claim id of the best-backed bullish thesis
    bear_thesis: str | None
    bull_sources: list             # distinct sources behind the bull thesis
    bear_sources: list
    contradictions: list           # [{"a": id, "b": id, "severity": str}]
    reason: str                    # one-line justification / abstention reason
    mechanism: str
    as_of: str | None
    min_margin: float
    min_sources: int
    min_source_reliability: float
    claims: list = field(default_factory=list)  # ranked per-claim breakdown

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_signal_corpus(path):
    """Read a signal corpus file into (asset, as_of, claims, relationships)."""
    data = json.loads(Path(path).read_text())
    return (data["asset"], data.get("as_of"),
            list(data["claims"]), list(data.get("relationships", [])))


def _validate_claims(claims):
    seen = set()
    for c in claims:
        for key in ("id", "claim_text", "source", "source_reliability", "stance"):
            if key not in c:
                raise ValueError(f"claim {c.get('id', '?')!r} is missing {key!r}")
        if c["stance"] not in STANCES:
            raise ValueError(
                f"claim {c['id']!r} has stance {c['stance']!r}; "
                f"expected one of {STANCES}")
        r = float(c["source_reliability"])
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"claim {c['id']!r} reliability {r} not in [0, 1]")
        if c["id"] in seen:
            raise ValueError(f"duplicate claim id {c['id']!r}")
        seen.add(c["id"])


def build_signal_graph(asset, claims, relationships):
    """Seed a ReasoningGraph exactly as the corpus loaders do."""
    _validate_claims(claims)
    g = ReasoningGraph(f"signal:{asset}")
    for c in claims:
        g.add_thought(
            c["id"], c["claim_text"],
            confidence=float(c["source_reliability"]),
            metadata={
                "source": c["source"],
                "source_reliability": float(c["source_reliability"]),
                "stance": c["stance"],
                "asset": asset,
                "doc_id": c.get("doc_id"),
            },
        )
    for rel in relationships:
        rel_type = _RELATION_MAP.get(rel["type"])
        if rel_type is None:
            raise ValueError(f"Unknown relationship type: {rel['type']!r}")
        edge_conf = g.graph.nodes[rel["from"]]["metadata"]["source_reliability"]
        g.add_relation(rel["from"], rel["to"], rel_type, confidence=edge_conf)

    # The engine's contradiction edges are DIRECTED: only the target node is
    # penalized. That is right for evidence -> thesis refutation (the evidence
    # is not weakened by the thesis it refutes) but wrong for two rival THESES,
    # which contradict each other symmetrically. Mirror thesis<->thesis
    # contradictions so neither side gets a free pass from edge direction.
    stance_of = {c["id"]: c["stance"] for c in claims}
    for rel in relationships:
        if rel["type"] != "contradiction":
            continue
        a, b = rel["from"], rel["to"]
        if (stance_of[a] in THESIS_STANCES and stance_of[b] in THESIS_STANCES
                and not g.graph.has_edge(b, a)):
            edge_conf = g.graph.nodes[b]["metadata"]["source_reliability"]
            g.add_relation(b, a, RelationType.CONTRADICTION, confidence=edge_conf,
                           metadata={"mirrored": True})
    return g


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def _backing_sources(g, node_id, min_reliability):
    """
    Distinct sources behind a thesis: its own source plus its supporters',
    counting only sources whose seed reliability clears `min_reliability`.
    A junk source (forum post, anonymous tip) may still add a little
    confidence through propagation, but it must not count as corroboration.
    """
    sources = set()
    own = g.graph.nodes[node_id]["metadata"]
    if own["source_reliability"] >= min_reliability:
        sources.add(own["source"])
    for pred in g.graph.predecessors(node_id):
        if g.graph[pred][node_id].get("type") in _SUPPORT_TYPES:
            meta = g.graph.nodes[pred]["metadata"]
            if meta["source_reliability"] >= min_reliability:
                sources.add(meta["source"])
    return sorted(sources)


def _best_thesis(g, stance):
    """(claim_id, final_conf) of the best-backed thesis on one side, or (None, 0)."""
    candidates = [(n, d["confidence"]) for n, d in g.graph.nodes(data=True)
                  if d["metadata"].get("stance") == stance]
    if not candidates:
        return None, 0.0
    # Deterministic tie-break on id so the call does not depend on node order.
    return max(candidates, key=lambda x: (x[1], x[0]))


def reason_signal(asset, claims, relationships, mechanism="logodds",
                  min_margin=0.05, min_sources=2, min_source_reliability=0.5,
                  as_of=None):
    """
    Run the contradiction engine over one asset's claims and return a
    Suggestion. See the module docstring for the decision rule.

    Parameters
    ----------
    mechanism : "logodds" (default) or "additive" — passed straight to
        ReasoningGraph.update_confidence_with_graph_structure.
    min_margin : minimum bull/bear score separation to make a call.
    min_sources : minimum distinct sources behind the winning thesis.
    min_source_reliability : a source only counts toward `min_sources` if its
        seed reliability is at least this (junk sources are not corroboration).
    """
    g = build_signal_graph(asset, claims, relationships)
    g.update_confidence_with_graph_structure(mechanism=mechanism)
    found = g.detect_contradictions()
    g.resolve_contradictions(found)
    # No pruning: a suggestion must surface everything it considered.

    bull_id, bull_score = _best_thesis(g, "bullish")
    bear_id, bear_score = _best_thesis(g, "bearish")
    margin = bull_score - bear_score

    bull_sources = (_backing_sources(g, bull_id, min_source_reliability)
                    if bull_id else [])
    bear_sources = (_backing_sources(g, bear_id, min_source_reliability)
                    if bear_id else [])

    # One row per unordered pair (a mirrored thesis<->thesis edge is one conflict).
    contradictions, seen_pairs = [], set()
    for a, b, _reason, sev in found:
        if a == b or frozenset((a, b)) in seen_pairs:
            continue
        seen_pairs.add(frozenset((a, b)))
        contradictions.append({"a": a, "b": b, "severity": sev})

    if bull_id is None and bear_id is None:
        action, reason = "no_call", "no thesis claims (only neutral evidence)"
    elif abs(margin) < min_margin:
        action = "no_call"
        reason = (f"margin {margin:+.3f} is below min_margin {min_margin}; "
                  f"the evidence does not separate the theses")
    else:
        winner, w_id, w_sources = (
            ("bull", bull_id, bull_sources) if margin > 0
            else ("bear", bear_id, bear_sources))
        if len(w_sources) < min_sources:
            action = "no_call"
            reason = (f"{winner} thesis {w_id} leads by {abs(margin):.3f} but is "
                      f"backed by {len(w_sources)} distinct source(s) "
                      f"(min {min_sources}); not corroborated")
        elif winner == "bull":
            action = "suggest_buy"
            reason = (f"bull thesis {bull_id} ({bull_score:.3f}) outranks bear "
                      f"thesis {bear_id or 'none'} ({bear_score:.3f}) by "
                      f"{margin:.3f}, backed by {len(bull_sources)} sources")
        else:
            action = "suggest_avoid"
            reason = (f"bear thesis {bear_id} ({bear_score:.3f}) outranks bull "
                      f"thesis {bull_id or 'none'} ({bull_score:.3f}) by "
                      f"{-margin:.3f}, backed by {len(bear_sources)} sources")

    ranked = sorted(g.graph.nodes(data=True),
                    key=lambda nd: (nd[1]["confidence"], nd[0]), reverse=True)
    claim_rows = [{
        "id": n,
        "stance": d["metadata"]["stance"],
        "source": d["metadata"]["source"],
        "seed": d["metadata"]["source_reliability"],
        "final": round(d["confidence"], 4),
        "support_count": d["metadata"].get("support_count", 0),
        "contradiction_count": d["metadata"].get("contradiction_count", 0),
        "text": d["content"],
    } for n, d in ranked]

    return Suggestion(
        asset=asset, action=action,
        bull_score=round(bull_score, 4), bear_score=round(bear_score, 4),
        margin=round(margin, 4),
        bull_thesis=bull_id, bear_thesis=bear_id,
        bull_sources=bull_sources, bear_sources=bear_sources,
        contradictions=contradictions, reason=reason,
        mechanism=mechanism, as_of=as_of,
        min_margin=min_margin, min_sources=min_sources,
        min_source_reliability=min_source_reliability,
        claims=claim_rows,
    )


def flat_baseline(claims):
    """
    "Trust the most reliable source": the stance of the highest-reliability
    THESIS claim, with no graph reasoning. Returns (stance, claim_id) or
    (None, None) when there are no thesis claims. This is the comparison
    point for reason_signal, mirroring experiments/baseline_vs_engine.py.
    """
    theses = [c for c in claims if c["stance"] in THESIS_STANCES]
    if not theses:
        return None, None
    best = max(theses, key=lambda c: (float(c["source_reliability"]), c["id"]))
    return best["stance"], best["id"]


# ---------------------------------------------------------------------------
# Decision log (for forward measurement)
# ---------------------------------------------------------------------------

def log_suggestion(path, suggestion, include_claims=False):
    """
    Append one Suggestion to a JSONL decision log. Records `logged_at` (UTC)
    alongside the suggestion's own `as_of`, so a later scorer can join each
    call to realized returns over its horizon. Returns the path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = suggestion.to_dict()
    if not include_claims:
        record.pop("claims", None)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def format_suggestion(s, top_n=8):
    """Human-readable rendering of a Suggestion for terminal/agent output."""
    lines = [
        f"{s.asset}: {s.action.upper()}   (as of {s.as_of or 'n/a'}, mechanism={s.mechanism})",
        f"  {s.reason}",
        f"  bull {s.bull_score:.3f} [{s.bull_thesis or '-'}]  vs  "
        f"bear {s.bear_score:.3f} [{s.bear_thesis or '-'}]  margin {s.margin:+.3f}",
        f"  bull sources: {', '.join(s.bull_sources) or '-'}",
        f"  bear sources: {', '.join(s.bear_sources) or '-'}",
    ]
    if s.contradictions:
        lines.append("  contradictions: " + "; ".join(
            f"{c['a']} <-> {c['b']} ({c['severity']})" for c in s.contradictions))
    lines.append("  ranked claims:")
    for row in s.claims[:top_n]:
        lines.append(f"    {row['final']:.3f} (seed {row['seed']:.2f}) "
                     f"{row['stance']:<8} {row['id']:<22} {row['source']}")
    return "\n".join(lines)
