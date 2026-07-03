"""
Contradiction handling for GraphRAG: contradictions are KEPT and TAGGED as
contested knowledge, not destroyed and pruned.

Old (trading-era) behavior: each contradiction penalized both sides by up to
-0.30, so a node in several conflicts was driven to ~0 and pruned — the
3-node standalone graph collapsed to 1 surviving node.

New behavior, proven here on the same 3-node graph:
  1. All conflicting nodes SURVIVE the full pipeline (confidence pass,
     detect, resolve, prune).
  2. Each carries a queryable `conflicts_with` metadata tag.
  3. Relative confidence ordering reflects relative reliability (the more
     confident node in a conflicting pair stays ranked above the other),
     and no node is driven below the conflict floor (0.15).
  4. Genuinely unsupported, conflict-free nodes are still pruned as before.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning import ReasoningGraph, RelationType

g = ReasoningGraph(
    "contradiction-handling",
    positive_keywords=['confirmed', 'verified'],
    negative_keywords=['disputed', 'retracted'],
)

# Same genuine contradiction as the standalone test: A is supported by B and
# contradicted by C, and A/B ('verified'/'confirmed') semantically oppose
# C ('disputed').
g.add_thought('A', 'Primary claim: the dataset is verified complete', confidence=0.8)
g.add_thought('B', 'Audit log confirmed all records present', confidence=0.7)
g.add_thought('C', 'Follow-up review disputed the record count', confidence=0.7)
g.add_relation('B', 'A', RelationType.SUPPORT, confidence=0.9)
g.add_relation('C', 'A', RelationType.CONTRADICTION, confidence=0.8)
assert len(g.graph.nodes) == 3

# --- Full pipeline ---------------------------------------------------------
g.update_confidence_with_graph_structure()
pre_resolution = {n: g.graph.nodes[n]['confidence'] for n in g.graph.nodes}

contradictions = g.detect_contradictions()
assert len(contradictions) >= 2, f"expected edge + semantic conflicts, got {contradictions}"

g.resolve_contradictions(contradictions)
removed = g.prune_low_confidence(threshold=0.5)

# --- 1. All 3 conflicting nodes survive (old behavior collapsed to 1) ------
surviving = sorted(g.graph.nodes)
assert surviving == ['A', 'B', 'C'], (
    f"conflicting nodes must survive pruning, got {surviving} "
    f"(rejected: {list(g.rejected_thoughts)})")
assert removed == 0, f"no conflicted node may be pruned, but {removed} were"

# --- 2. Every conflicting node carries a conflicts_with tag ----------------
tags = {}
for n in surviving:
    conflicts = g.graph.nodes[n]['metadata'].get('conflicts_with')
    assert conflicts, f"node {n} is in a contradiction but has no conflicts_with tag"
    for entry in conflicts:
        assert entry['node'] in ('A', 'B', 'C') and entry['severity']
    tags[n] = [(e['node'], e['severity']) for e in conflicts]

# --- 3. Relative ordering preserved; nobody nuked to ~0 --------------------
post = {n: g.graph.nodes[n]['confidence'] for n in surviving}
assert pre_resolution['A'] > pre_resolution['C'], "test premise: A entered resolution more reliable"
assert post['A'] > post['C'], (
    f"more reliable node must stay ranked above its contradictor: {post}")
for n, conf in post.items():
    assert conf >= 0.15, f"{n} driven below the conflict floor: {conf:.3f}"

# Floor holds even under repeated resolution passes (old code hit 0 in one pass).
for _ in range(50):
    g.resolve_contradictions(contradictions)
for n in surviving:
    assert g.graph.nodes[n]['confidence'] >= 0.15, \
        f"{n} fell below floor after repeated resolution"

# --- 4. Conflict-free junk is still pruned as before -----------------------
g.add_thought('D', 'Unrelated stray note with no backing', confidence=0.2)
assert g.prune_low_confidence(threshold=0.5) == 1 and 'D' in g.rejected_thoughts, \
    "genuinely weak, conflict-free nodes must still be prunable"
assert sorted(g.graph.nodes) == ['A', 'B', 'C']

# --- Report -----------------------------------------------------------------
print("OLD behavior: 3 nodes -> 1 (destroyed). "
      f"NEW behavior: 3 nodes -> {len(g.graph.nodes)} (kept + tagged).")
print(f"surviving nodes: {sorted(g.graph.nodes)}")
for n in sorted(g.graph.nodes):
    print(f"  {n}: confidence {pre_resolution[n]:.3f} -> {post[n]:.3f}, "
          f"conflicts_with={tags[n]}")
print("conflict-free node D pruned as before: "
      f"rejected_thoughts={list(g.rejected_thoughts)}")

print("\nCONTRADICTION HANDLING TEST: PASS")
