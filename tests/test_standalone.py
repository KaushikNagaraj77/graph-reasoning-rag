"""
Standalone smoke test: the reasoning engine must work with zero trading code.
Builds a tiny 3-node graph, runs the full pipeline (confidence pass,
contradiction detection/resolution, pruning, reconsideration), asserts each
stage behaves.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning import ReasoningGraph, RelationType

g = ReasoningGraph(
    "smoke-test",
    positive_keywords=['confirmed', 'verified'],
    negative_keywords=['disputed', 'retracted'],
)

# 3 nodes: a root claim, a supporting claim, and a contradicting claim
g.add_thought('A', 'Primary claim: the dataset is verified complete', confidence=0.8)
g.add_thought('B', 'Audit log confirmed all records present', confidence=0.7)
g.add_thought('C', 'Follow-up review disputed the record count', confidence=0.7)

# 2 edges: B supports A, C contradicts A
g.add_relation('B', 'A', RelationType.SUPPORT, confidence=0.9)
g.add_relation('C', 'A', RelationType.CONTRADICTION, confidence=0.8)

assert len(g.graph.nodes) == 3 and len(g.graph.edges) == 2

# 1. Structure-based confidence pass
before_a = g.graph.nodes['A']['confidence']
g.update_confidence_with_graph_structure()
after_a = g.graph.nodes['A']['confidence']
meta_a = g.graph.nodes['A']['metadata']
assert meta_a['support_count'] == 1, "A should see one supporting edge"
assert meta_a['contradiction_count'] == 1, "A should see one contradicting edge"
print(f"1. confidence pass: A {before_a:.3f} -> {after_a:.3f} "
      f"(support={meta_a['support_count']}, contra={meta_a['contradiction_count']})")

# 2. Contradiction detection: edge-based (C->A, both >0.6) AND
#    semantic ('verified/confirmed' vs 'disputed')
contradictions = g.detect_contradictions()
assert len(contradictions) >= 2, f"expected edge + semantic contradictions, got {contradictions}"
print(f"2. contradictions detected: {len(contradictions)}")
for a, b, reason, sev in contradictions:
    print(f"   {a} <-> {b} [{sev}] {reason}")

# 3. Resolution penalizes both sides
conf_c_before = g.graph.nodes['C']['confidence']
g.resolve_contradictions(contradictions)
conf_c_after = g.graph.nodes['C']['confidence']
assert conf_c_after < conf_c_before, "C should lose confidence in resolution"
print(f"3. resolution: C {conf_c_before:.3f} -> {conf_c_after:.3f}")

# 4. Pruning moves weak nodes to rejected_thoughts
removed = g.prune_low_confidence(threshold=0.5)
assert removed >= 1, "at least one penalized node should fall below 0.5"
assert len(g.rejected_thoughts) == removed
print(f"4. pruned {removed} node(s) -> rejected_thoughts: {list(g.rejected_thoughts)}")

# 5. Reconsideration can readmit (20% bump over 0.4 threshold)
readded = g.reconsider_rejected_thoughts(confidence_threshold=0.4)
print(f"5. readmitted {readded} thought(s); graph now has {len(g.graph.nodes)} nodes")

# 6. No trading modules anywhere in the import graph
banned = [m for m in sys.modules
          if any(t in m for t in ('yfinance', 'talib', 'torch', 'backtest',
                                  'rl_exit_agent', 'data_fetch', 'matplotlib', 'pandas'))]
assert not banned, f"trading/heavy deps leaked into import graph: {banned}"
print(f"6. import graph clean: no trading dependencies loaded")

print("\nSTANDALONE TEST: PASS")
