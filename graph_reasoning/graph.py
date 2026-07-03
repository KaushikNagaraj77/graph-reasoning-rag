"""
ReasoningGraph — a contradiction-aware, confidence-weighted reasoning graph.

Extracted from the Binance_TradeX project's TradingThoughtGraph
(utils/analysis/graph.py) and made domain-agnostic. Core ideas preserved:

  - Thoughts are nodes with a content string and a confidence score [0, 1].
  - Typed, directed edges encode logical relationships (support,
    contradiction, confluence, derivation, ...).
  - Confidence is refined from GRAPH STRUCTURE: well-supported nodes gain
    confidence, contradicted nodes lose it, confluence amplifies.
  - Contradictions are detected (edge-based and, optionally, keyword-based),
    tagged on both nodes' metadata (`conflicts_with`), and resolved by a
    small relative-reliability re-ranking. Contested knowledge is KEPT and
    surfaced, not destroyed (the source project penalized both sides toward
    0 and pruned them — wrong for a knowledge/RAG system).
  - Low-confidence nodes are pruned into `rejected_thoughts` (memory
    retention) and may later be readmitted by `reconsider_rejected_thoughts`;
    conflict-tagged nodes are protected from pruning.

Changes from the original (see docs/PRIOR_PROJECT_FINDINGS.md for the source
project's lessons):
  - All trading dependencies stripped: yfinance, pandas, matplotlib,
    talib_wrapper, the .visualization module, and the deprecated
    torch/GNN pathway (the source project itself disabled it: graph-structure
    confidence replaced it).
  - Semantic (keyword) contradiction detection is now parameterized:
    pass `positive_keywords` / `negative_keywords` to the constructor for
    your domain (the original hardcoded bullish/bearish trading terms).
    With no keywords supplied, edge-based contradiction detection still works.
  - `visualize()` removed (depended on the trading repo's viz module).

Dependencies: networkx, numpy.
"""

import random
from datetime import datetime
from enum import Enum

import networkx as nx
import numpy as np


class RelationType(Enum):
    """Types of logical relationships between thoughts."""
    SUPPORT = "support"                  # Confirms or supports a conclusion
    CONTRADICTION = "contradiction"      # Contradicts a conclusion
    INDICATION = "indication"            # Indicates a possibility
    CONFLUENCE = "confluence"            # Multiple factors align
    PATTERN = "pattern"                  # Pattern recognition
    DERIVATION = "derivation"            # Derived from calculation
    VALID_INFERENCE = "valid_inference"  # Logical inference
    FALLACY = "fallacy"                  # Logical fallacy
    EXPANSION = "expansion"              # Thought expansion
    TIMEFRAME_ALIGNMENT = "timeframe_alignment"  # Cross-context alignment


class ReasoningGraph:
    """Confidence-weighted graph of thoughts with contradiction resolution."""

    def __init__(self, name="Reasoning Graph",
                 positive_keywords=None, negative_keywords=None):
        """
        Parameters
        ----------
        name : str
            Graph label.
        positive_keywords / negative_keywords : list[str] or None
            Optional domain vocabulary for SEMANTIC contradiction detection:
            a high-confidence node whose content matches one polarity
            contradicts a high-confidence node matching the other.
            (The source project used bullish/bearish trading terms here.)
            If omitted, only explicit CONTRADICTION edges are detected.
        """
        self.name = name
        self.graph = nx.DiGraph(name=name)
        self.rejected_thoughts = {}          # memory retention for pruned thoughts
        self.graph_structure_influence = {}  # metrics from the last confidence pass
        self.creation_counter = 0            # node creation order
        self.positive_keywords = list(positive_keywords or [])
        self.negative_keywords = list(negative_keywords or [])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_thought(self, id, content, confidence=0.5, metadata=None):
        """Add (or update) a thought node."""
        if metadata is None:
            metadata = {}

        if id in self.graph:
            self.graph.nodes[id]['content'] = content
            self.graph.nodes[id]['confidence'] = confidence
            if metadata:
                if 'metadata' not in self.graph.nodes[id]:
                    self.graph.nodes[id]['metadata'] = {}
                self.graph.nodes[id]['metadata'].update(metadata)
        else:
            self.graph.add_node(
                id,
                content=content,
                confidence=confidence,
                type="thought",
                creation_time=datetime.now(),
                creation_order=self.creation_counter,
                metadata=dict(metadata),
            )
            self.creation_counter += 1
        return id

    def add_relation(self, from_id, to_id, relation_type, confidence=0.5, metadata=None):
        """Add a directed, typed edge between two existing thoughts."""
        if metadata is None:
            metadata = {}
        if from_id not in self.graph or to_id not in self.graph:
            raise ValueError(f"One or both nodes not found: {from_id}, {to_id}")
        self.graph.add_edge(
            from_id, to_id,
            type=relation_type.value,
            confidence=confidence,
            metadata=metadata,
        )
        return (from_id, to_id)

    # ------------------------------------------------------------------
    # Structure-based confidence
    # ------------------------------------------------------------------

    def _count_support_edges(self, node_id):
        """Count incoming supportive edges and their confidence mass."""
        support_count = 0
        support_confidence = 0.0
        for pred in self.graph.predecessors(node_id):
            edge = self.graph[pred][node_id]
            if edge.get('type', '') in (RelationType.SUPPORT.value,
                                        RelationType.CONFLUENCE.value,
                                        RelationType.VALID_INFERENCE.value,
                                        RelationType.DERIVATION.value):
                support_count += 1
                support_confidence += (edge.get('confidence', 0.5) *
                                       self.graph.nodes[pred].get('confidence', 0.5))
        return support_count, support_confidence

    def _count_contradiction_edges(self, node_id):
        """Count incoming contradicting edges and their confidence mass."""
        contradiction_count = 0
        contradiction_confidence = 0.0
        for pred in self.graph.predecessors(node_id):
            edge = self.graph[pred][node_id]
            if edge.get('type', '') in (RelationType.CONTRADICTION.value,
                                        RelationType.FALLACY.value):
                contradiction_count += 1
                contradiction_confidence += (edge.get('confidence', 0.5) *
                                             self.graph.nodes[pred].get('confidence', 0.5))
        return contradiction_count, contradiction_confidence

    def update_confidence_with_graph_structure(self, verbose=False):
        """
        Refine node confidences from graph topology:
          - supporting edges boost (diminishing, capped at +0.4)
          - contradiction edges penalize (capped at -0.4)
          - each incoming CONFLUENCE edge adds a further +0.10
        Results are clamped to [0, 1]; per-node deltas are stored in node
        metadata and aggregate metrics in `self.graph_structure_influence`.
        """
        if len(self.graph.nodes) < 2:
            return

        confidence_changes = {}
        for node_id in self.graph.nodes:
            original = self.graph.nodes[node_id].get('confidence', 0.5)
            support_count, support_conf = self._count_support_edges(node_id)
            contra_count, contra_conf = self._count_contradiction_edges(node_id)

            new_confidence = original
            if support_count > 0:
                new_confidence += min(0.4, support_count * 0.15 *
                                      (support_conf / max(support_count, 1)))
            if contra_count > 0:
                new_confidence -= min(0.4, contra_count * 0.20 *
                                      (contra_conf / max(contra_count, 1)))

            confluence_boost = sum(
                0.10 for pred in self.graph.predecessors(node_id)
                if self.graph[pred][node_id].get('type', '') == RelationType.CONFLUENCE.value
            )
            new_confidence = max(0.0, min(1.0, new_confidence + confluence_boost))

            self.graph.nodes[node_id]['confidence'] = new_confidence
            confidence_changes[node_id] = new_confidence - original

            meta = self.graph.nodes[node_id].setdefault('metadata', {})
            meta['support_count'] = support_count
            meta['contradiction_count'] = contra_count
            meta['confidence_change'] = new_confidence - original

        self.graph_structure_influence = {
            'confidence_changes': confidence_changes,
            'avg_abs_change': float(np.mean([abs(c) for c in confidence_changes.values()]))
                              if confidence_changes else 0.0,
            'max_increase': max(confidence_changes.values()) if confidence_changes else 0.0,
            'max_decrease': min(confidence_changes.values()) if confidence_changes else 0.0,
            'nodes_boosted': sum(1 for c in confidence_changes.values() if c > 0.05),
            'nodes_penalized': sum(1 for c in confidence_changes.values() if c < -0.05),
        }
        if verbose:
            gi = self.graph_structure_influence
            print(f"[{self.name}] confidence pass: avg|Δ|={gi['avg_abs_change']:.3f} "
                  f"boosted={gi['nodes_boosted']} penalized={gi['nodes_penalized']}")

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def detect_contradictions(self, confidence_threshold=0.5, verbose=False):
        """
        Detect contradictions two ways:
          1. Edge-based: an explicit CONTRADICTION edge between two nodes that
             BOTH hold confidence > 0.6 (severity: high).
          2. Semantic (only if polarity keywords were provided): every pair of
             high-confidence nodes on opposite polarities contradicts
             (severity: high); a single node matching both polarities
             contradicts itself (severity: medium).

        Returns a list of (node_a, node_b, reason, severity) tuples.
        """
        contradictions = []
        positive_nodes, negative_nodes = [], []

        for node in self.graph.nodes:
            content = self.graph.nodes[node].get('content', '').lower()
            conf = self.graph.nodes[node].get('confidence', 0.5)

            for pred in self.graph.predecessors(node):
                if self.graph[pred][node].get('type', '') == RelationType.CONTRADICTION.value:
                    pred_conf = self.graph.nodes[pred].get('confidence', 0.5)
                    if conf > 0.6 and pred_conf > 0.6:
                        contradictions.append((
                            pred, node,
                            "Both nodes have high confidence but contradict each other",
                            "high"))

            if self.positive_keywords or self.negative_keywords:
                is_pos = any(kw in content for kw in self.positive_keywords)
                is_neg = any(kw in content for kw in self.negative_keywords)
                if is_pos:
                    positive_nodes.append((node, conf))
                if is_neg:
                    negative_nodes.append((node, conf))
                if is_pos and is_neg:
                    contradictions.append((
                        node, node,
                        "Node contains both polarities of the domain vocabulary",
                        "medium"))

        for pos_node, pos_conf in positive_nodes:
            if pos_conf > confidence_threshold:
                for neg_node, neg_conf in negative_nodes:
                    if neg_node != pos_node and neg_conf > confidence_threshold:
                        contradictions.append((
                            pos_node, neg_node,
                            f"High-confidence opposite-polarity nodes "
                            f"({pos_conf:.2f} vs {neg_conf:.2f})",
                            "high"))

        # Record each conflict on BOTH nodes' metadata so contested knowledge
        # stays queryable at retrieval time (kept and surfaced, not destroyed).
        for node1, node2, reason, severity in contradictions:
            for node, other in ((node1, node2), (node2, node1)):
                if node not in self.graph.nodes:
                    continue
                conflicts = (self.graph.nodes[node]
                             .setdefault('metadata', {})
                             .setdefault('conflicts_with', []))
                if not any(c.get('node') == other and c.get('severity') == severity
                           for c in conflicts):
                    conflicts.append({'node': other,
                                      'severity': severity,
                                      'reason': reason})

        if verbose:
            print(f"[{self.name}] contradictions found: {len(contradictions)}")
        return contradictions

    def resolve_contradictions(self, contradictions=None, verbose=False,
                               min_confidence=0.15):
        """
        Re-rank both sides of each contradiction instead of destroying them.
        A small penalty (high -0.10, medium -0.06, low -0.03) is split by
        relative reliability — the less confident node absorbs the larger
        share — so the more reliable node stays ranked above the other, and
        neither is driven toward 0. Node confidence (the reliability proxy
        for now) never drops below `min_confidence` purely due to conflict.
        Returns the number of contradictions resolved.
        """
        if contradictions is None:
            contradictions = self.detect_contradictions()
        if not contradictions:
            return 0

        penalty_map = {'high': 0.10, 'medium': 0.06, 'low': 0.03}
        for node1, node2, reason, severity in contradictions:
            base_penalty = penalty_map.get(severity, 0.06)
            confs = {node: self.graph.nodes[node].get('confidence', 0.5)
                     for node in {node1, node2} if node in self.graph.nodes}
            total = sum(confs.values())
            for node, old in confs.items():
                if len(confs) == 2 and total > 0:
                    share = (total - old) / total  # weaker side absorbs more
                else:
                    share = 0.5  # self-contradiction or degenerate pair
                new = max(min(old, min_confidence), old - base_penalty * share)
                self.graph.nodes[node]['confidence'] = new
                if verbose:
                    print(f"  {node}: {old:.3f} -> {new:.3f} ({reason})")
        return len(contradictions)

    # ------------------------------------------------------------------
    # Pruning and reconsideration
    # ------------------------------------------------------------------

    def apply_logical_pruning(self, fallacy_markers=("fallacy", "invalid", "error")):
        """
        Remove nodes whose content contains any fallacy marker, retaining them
        in `rejected_thoughts`. Markers are parameterized (the original
        hardcoded trading-specific terms alongside these).
        """
        nodes_to_remove = []
        for node in self.graph.nodes:
            content = self.graph.nodes[node].get('content', '').lower()
            if any(marker in content for marker in fallacy_markers):
                nodes_to_remove.append(node)
                self.rejected_thoughts[node] = {
                    'content': self.graph.nodes[node].get('content', ''),
                    'reason': 'Fallacy detected',
                    'confidence': self.graph.nodes[node].get('confidence', 0),
                }
        self.graph.remove_nodes_from(nodes_to_remove)
        return len(nodes_to_remove)

    def prune_low_confidence(self, threshold=0.5):
        """Move nodes below the confidence threshold into rejected_thoughts.

        Contested nodes (non-empty `conflicts_with` metadata) are never
        pruned here: a conflict marks valuable knowledge that must remain
        available to surface at query time. Only genuinely unsupported,
        conflict-free nodes are removed.
        """
        nodes_to_remove = [n for n, d in self.graph.nodes(data=True)
                           if d.get('confidence', 0) < threshold
                           and not d.get('metadata', {}).get('conflicts_with')]
        for node in nodes_to_remove:
            self.rejected_thoughts[node] = {
                'content': self.graph.nodes[node].get('content', ''),
                'reason': 'Low confidence',
                'confidence': self.graph.nodes[node].get('confidence', 0),
            }
        self.graph.remove_nodes_from(nodes_to_remove)
        return len(nodes_to_remove)

    def reconsider_rejected_thoughts(self, confidence_threshold=0.4):
        """
        Give rejected thoughts a second chance: bump confidence by 20% and
        readmit any that clear the threshold, reconnecting them to the most
        content-similar surviving nodes (word-overlap heuristic).
        Returns the number of readmitted thoughts.
        """
        if not self.rejected_thoughts:
            return 0

        readded = 0
        for node_id, data in list(self.rejected_thoughts.items()):
            new_confidence = min(0.95, data.get('confidence', 0) * 1.2)
            if new_confidence < confidence_threshold:
                continue

            new_id = f"readmitted_{node_id}_{len(self.graph.nodes)}"
            content = data.get('content', '')
            self.add_thought(new_id, content, confidence=new_confidence,
                             metadata={'readmitted': True, 'original_id': node_id})
            readded += 1

            # Reconnect to content-similar nodes
            content_words = {w.lower() for w in content.split() if len(w) > 4}
            relevant = []
            for existing in list(self.graph.nodes):
                if existing == new_id:
                    continue
                existing_words = {w.lower()
                                  for w in self.graph.nodes[existing]['content'].split()
                                  if len(w) > 4}
                if content_words and existing_words:
                    overlap = (len(content_words & existing_words) /
                               len(content_words | existing_words))
                    if overlap > 0.1:
                        relevant.append((existing, overlap))
            relevant.sort(key=lambda x: x[1], reverse=True)

            connected = False
            for node, relevance in relevant[:3]:
                rel = RelationType.SUPPORT if relevance > 0.3 else RelationType.INDICATION
                if random.random() < 0.7:
                    self.add_relation(node, new_id, rel, confidence=0.5 + relevance / 2)
                    connected = True

            if not connected and len(self.graph.nodes) > 1:
                by_conf = sorted(((n, self.graph.nodes[n].get('confidence', 0))
                                  for n in self.graph.nodes if n != new_id),
                                 key=lambda x: x[1], reverse=True)
                if by_conf:
                    self.add_relation(by_conf[0][0], new_id,
                                      RelationType.EXPANSION, confidence=0.6)

            del self.rejected_thoughts[node_id]
        return readded

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def find_optimal_path(self):
        """
        Find the root->leaf path with the highest product of node and edge
        confidences (longer paths get a mild bonus). Returns (path, confidence).
        """
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        leafs = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        if not leafs:
            leafs = [n for n in self.graph.nodes
                     if self.graph.nodes[n]['confidence'] > 0.7
                     and self.graph.out_degree(n) > 0]

        best_path, best_confidence = None, 0
        for root in roots:
            for leaf in leafs:
                try:
                    for path in nx.all_simple_paths(self.graph, root, leaf):
                        confidence = 1.0
                        for node in path:
                            confidence *= self.graph.nodes[node].get('confidence', 0.5)
                        for i in range(len(path) - 1):
                            confidence *= self.graph[path[i]][path[i + 1]].get('confidence', 0.5)
                        confidence *= min(1.2, 1 + (len(path) / 20))
                        if confidence > best_confidence:
                            best_confidence, best_path = confidence, path
                except nx.NetworkXError:
                    continue
        return best_path, best_confidence


# Backward-compatible alias for code ported from the source project.
TradingThoughtGraph = ReasoningGraph
