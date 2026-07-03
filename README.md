# graph-reasoning-rag

A contradiction-aware, confidence-weighted GraphRAG built on a reasoning-graph
engine extracted from a prior trading-research project.

## Goal

Retrieval-augmented generation where retrieved evidence is organized as a
**reasoning graph** rather than a flat context window:

- Each retrieved claim/passage becomes a **thought node** with a confidence score.
- Typed edges encode logical relationships between claims (support,
  contradiction, derivation, confluence, ...).
- **Graph-structure confidence propagation** boosts well-supported claims and
  penalizes contradicted ones before anything reaches the generator.
- **Contradiction detection/resolution** surfaces conflicting evidence
  explicitly instead of letting the generator silently average over it.
- Pruned low-confidence claims are retained (`rejected_thoughts`) and can be
  readmitted when new evidence arrives.

## The engine

`graph_reasoning/graph.py` — `ReasoningGraph` (domain-agnostic). Core API:

| Method | Purpose |
|---|---|
| `add_thought(id, content, confidence)` | Add a claim node |
| `add_relation(a, b, RelationType, confidence)` | Typed directed edge |
| `update_confidence_with_graph_structure()` | Structure-based confidence pass |
| `detect_contradictions()` / `resolve_contradictions()` | Find + penalize conflicts |
| `prune_low_confidence(threshold)` | Move weak nodes to memory |
| `reconsider_rejected_thoughts()` | Readmit rescued thoughts |
| `find_optimal_path()` | Highest-confidence reasoning chain |

Semantic contradiction detection is vocabulary-driven: pass
`positive_keywords` / `negative_keywords` for your domain at construction.

## Lessons carried over from the source project

See `docs/PRIOR_PROJECT_FINDINGS.md` (the full research log). The three that
shape this project:

1. **Dynamic beats static**: a graph built once over all data produced a
   frozen, useless verdict; rebuilt on trailing windows it tracked regime
   changes. Rebuild the graph as evidence changes.
2. **Contradiction resolution is the engine's real value**: keyword-classified
   node polarity + confidence penalties changed downstream behavior measurably.
3. **Benchmark against trivial baselines, honestly**: the source project's
   headline numbers evaporated when measured against costs and baselines.
   Every RAG-quality claim here gets a dumb-baseline comparison first.

## Setup

```bash
pip install -r requirements.txt
python tests/test_standalone.py
```
