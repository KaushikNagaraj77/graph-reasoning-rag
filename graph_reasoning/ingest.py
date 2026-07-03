"""
Minimal ingestion layer for the contradiction-handling experiment.

Loads a small, hand-authored corpus of *claims* (each with a source and a
source-reliability score) plus explicit support/contradiction *relationships*
into a ReasoningGraph, so the existing contradiction pipeline runs on real
conflicting content. This is deliberately minimal scaffolding for a controlled
research experiment — not a general ingestion system.

Claim schema (see data/sample_corpus.json):
    id                 : str   unique node id
    claim_text         : str   the claim's content
    source             : str   human-readable source name
    source_reliability : float [0, 1]; seeds node confidence
    topic              : str   optional grouping label (carried in metadata)
    ground_truth       : bool  optional; True marks the known-correct claim

Relationship schema:
    from, to : claim ids
    type     : "support" | "contradiction" (mapped to RelationType)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .graph import ReasoningGraph, RelationType

_RELATION_MAP = {
    "support": RelationType.SUPPORT,
    "contradiction": RelationType.CONTRADICTION,
}


@dataclass
class Claim:
    """A single sourced claim to be loaded as a graph node."""
    id: str
    claim_text: str
    source: str
    source_reliability: float
    topic: str = ""
    ground_truth: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            claim_text=d["claim_text"],
            source=d["source"],
            source_reliability=float(d["source_reliability"]),
            topic=d.get("topic", ""),
            ground_truth=bool(d.get("ground_truth", False)),
        )


def load_corpus_file(path):
    """Read a JSON corpus file into (claims, relationships)."""
    data = json.loads(Path(path).read_text())
    claims = [Claim.from_dict(c) for c in data.get("claims", [])]
    relationships = list(data.get("relationships", []))
    return claims, relationships


def load_into_graph(claims, relationships, graph=None, name="corpus"):
    """
    Populate a ReasoningGraph from claims and relationships.

    Each claim becomes a node whose confidence is initialized from its
    source_reliability; source/topic/ground_truth are stored in node metadata
    so they survive the pipeline and can be inspected afterward. Each
    relationship becomes a typed edge via add_relation, weighted by the
    reliability of the source claim.

    Returns the populated graph.
    """
    if graph is None:
        graph = ReasoningGraph(name)

    for claim in claims:
        graph.add_thought(
            claim.id,
            claim.claim_text,
            confidence=claim.source_reliability,
            metadata={
                "source": claim.source,
                "source_reliability": claim.source_reliability,
                "topic": claim.topic,
                "ground_truth": claim.ground_truth,
            },
        )

    for rel in relationships:
        rel_type = _RELATION_MAP.get(rel["type"])
        if rel_type is None:
            raise ValueError(f"Unknown relationship type: {rel['type']!r}")
        from_id, to_id = rel["from"], rel["to"]
        edge_conf = graph.graph.nodes[from_id]["metadata"].get(
            "source_reliability", 0.5)
        graph.add_relation(from_id, to_id, rel_type, confidence=edge_conf)

    return graph
