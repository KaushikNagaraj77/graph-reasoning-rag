"""
Head-to-head: a flat reliability baseline vs. the full ReasoningGraph engine,
on the same 8-topic corpus.

Both rankers answer the same question per topic: among the *competing claims*
(the ground-truth claim and any rival claim it contradicts), which one wins?
Corroborating evidence nodes (support-only, no contradiction edge) are not
candidate answers for either system; they are the extra signal the engine may
exploit and the baseline ignores.

  BASELINE  — pick the competing claim with the highest SEED source_reliability.
              No graph, no support propagation, no contradiction handling.
  ENGINE    — run the full pipeline (structure confidence pass -> detect ->
              resolve -> prune) and pick the competing claim with the highest
              FINAL confidence.

Nothing is tuned; both run on data/sample_corpus.json as-is.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning import load_corpus_file, load_into_graph

CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.json"


def competing_claims(claims, relationships):
    """
    Per topic, the candidate answers = ground-truth claim + any claim that is an
    endpoint of a contradiction edge (the rival). Support-only evidence nodes
    are excluded. Returns {topic: [claim_id, ...]}.
    """
    contradiction_ids = set()
    for r in relationships:
        if r["type"] == "contradiction":
            contradiction_ids.add(r["from"])
            contradiction_ids.add(r["to"])

    topics = {}
    for c in claims:
        if c.ground_truth or c.id in contradiction_ids:
            topics.setdefault(c.topic, []).append(c.id)
    return topics


def main():
    claims, relationships = load_corpus_file(CORPUS)
    by_id = {c.id: c for c in claims}
    truth_ids = {c.id for c in claims if c.ground_truth}
    reliability = {c.id: c.source_reliability for c in claims}
    topic_contenders = competing_claims(claims, relationships)

    # --- BASELINE: highest seed reliability among contenders ---------------
    baseline_pick = {
        t: max(ids, key=lambda n: reliability[n])
        for t, ids in topic_contenders.items()
    }

    # --- ENGINE: full pipeline, then highest final confidence -------------
    g = load_into_graph(claims, relationships, name="head-to-head")
    g.update_confidence_with_graph_structure()
    contradictions = g.detect_contradictions()
    g.resolve_contradictions(contradictions)
    g.prune_low_confidence(threshold=0.5)

    def final_conf(n):
        # A contender pruned out of the graph loses by definition.
        if n in g.graph.nodes:
            return g.graph.nodes[n]["confidence"]
        return float("-inf")

    engine_pick = {
        t: max(ids, key=final_conf)
        for t, ids in topic_contenders.items()
    }

    # --- Report ------------------------------------------------------------
    hard = {"continental_drift", "chronic_gastritis", "puerperal_fever"}

    print("=" * 100)
    print("BASELINE (flat seed reliability)  vs  ENGINE (support-structure propagation)")
    print("=" * 100)
    header = (f"{'Topic':<20} {'Ground truth':<22} "
              f"{'Baseline pick':<24} {'Engine pick':<24}")
    print(header)
    print("-" * 100)

    b_hits = e_hits = 0
    rows = []
    for topic in sorted(topic_contenders):
        truth_id = next(iter(truth_ids & set(topic_contenders[topic])))
        b = baseline_pick[topic]
        e = engine_pick[topic]
        b_ok = b == truth_id
        e_ok = e == truth_id
        b_hits += b_ok
        e_hits += e_ok
        rows.append((topic, truth_id, b, b_ok, e, e_ok))
        tag = "  <-- HARD" if topic in hard else ""
        print(f"{topic:<20} {truth_id:<22} "
              f"{b + (' OK' if b_ok else ' WRONG'):<24} "
              f"{e + (' OK' if e_ok else ' WRONG'):<24}{tag}")

    print("-" * 100)
    print(f"TOTALS:  Baseline {b_hits}/{len(topic_contenders)}   "
          f"Engine {e_hits}/{len(topic_contenders)}")

    # --- Numeric backing so the head-to-head is hand-verifiable -----------
    print("\n" + "=" * 100)
    print("NUMERIC DETAIL (seed reliability drives baseline; final confidence drives engine)")
    print("=" * 100)
    for topic in sorted(topic_contenders):
        truth_id = next(iter(truth_ids & set(topic_contenders[topic])))
        print(f"\n{topic}{'   <-- HARD (correct claim has LOWER seed reliability)' if topic in hard else ''}")
        for n in sorted(topic_contenders[topic],
                        key=lambda x: reliability[x], reverse=True):
            gt = " [GT]" if n in truth_ids else "    "
            fc = final_conf(n)
            fc_txt = f"{fc:.4f}" if fc != float("-inf") else "pruned"
            print(f"   {gt} {n:<24} seed={reliability[n]:.2f}   final_conf={fc_txt}")

    # --- Hard-topic call-out ----------------------------------------------
    print("\n" + "=" * 100)
    print("HARD TOPICS — does support structure overturn the higher-reliability wrong claim?")
    print("=" * 100)
    for topic in sorted(hard):
        truth_id = next(iter(truth_ids & set(topic_contenders[topic])))
        rivals = [n for n in topic_contenders[topic] if n != truth_id]
        rival = max(rivals, key=lambda n: reliability[n])
        b = baseline_pick[topic]
        e = engine_pick[topic]
        print(f"\n{topic}:")
        print(f"   ground truth : {truth_id}  seed={reliability[truth_id]:.2f}  "
              f"final={final_conf(truth_id):.4f}")
        print(f"   rival (wrong): {rival}  seed={reliability[rival]:.2f}  "
              f"final={final_conf(rival):.4f}")
        print(f"   baseline picks {b} ({'CORRECT' if b == truth_id else 'WRONG — takes the high-reliability rival'})")
        verdict = ("engine OVERCOMES it (support lifts the correct claim above the rival)"
                   if e == truth_id else
                   "engine ALSO WRONG (support not enough to overcome seed gap)")
        print(f"   engine   picks {e} -> {verdict}")

    return b_hits, e_hits, len(topic_contenders)


if __name__ == "__main__":
    main()
