"""
Contradiction-handling experiment on a hand-authored, sourced corpus.

Loads data/sample_corpus.json into a ReasoningGraph, runs the full pipeline
(confidence pass -> detect -> resolve -> prune), then reports:
  - which claims survived and their conflict tags,
  - the final confidence ranking,
  - per conflict topic, whether the highest-confidence SURVIVING claim is the
    one flagged ground_truth.

The question this answers: when sources conflict, does the engine's
confidence ranking put the ground-truth claim on top?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning import RelationType, load_corpus_file, load_into_graph

CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.json"


def main():
    claims, relationships = load_corpus_file(CORPUS)
    truth_ids = {c.id for c in claims if c.ground_truth}
    topic_of = {c.id: c.topic for c in claims}
    source_of = {c.id: c.source for c in claims}
    reliability_of = {c.id: c.source_reliability for c in claims}  # baseline signal

    g = load_into_graph(claims, relationships, name="medical-claims")
    print(f"Loaded {len(claims)} claims across "
          f"{len(set(topic_of.values()))} topics, "
          f"{len(relationships)} relationships.\n")

    # --- Full pipeline -----------------------------------------------------
    g.update_confidence_with_graph_structure()
    contradictions = g.detect_contradictions()
    g.resolve_contradictions(contradictions)
    pruned = g.prune_low_confidence(threshold=0.5)

    survivors = list(g.graph.nodes)
    print(f"Pipeline: detected {len(contradictions)} contradiction pair(s); "
          f"pruned {pruned} conflict-free low-confidence claim(s).")
    print(f"Survivors: {len(survivors)}/{len(claims)} claims kept.\n")

    # --- Survivors, tags, and confidence ranking ---------------------------
    print("=" * 72)
    print("SURVIVING CLAIMS (ranked by final confidence)")
    print("=" * 72)
    ranked = sorted(survivors,
                    key=lambda n: g.graph.nodes[n]["confidence"],
                    reverse=True)
    for n in ranked:
        node = g.graph.nodes[n]
        meta = node["metadata"]
        conflicts = meta.get("conflicts_with", [])
        tag = ", ".join(f"{c['node']}({c['severity']})" for c in conflicts) or "-"
        gt = " [GROUND TRUTH]" if meta.get("ground_truth") else ""
        print(f"  {node['confidence']:.3f}  {n:<26} {topic_of[n]:<18}{gt}")
        print(f"         source: {source_of[n]}")
        print(f"         conflicts_with: {tag}")

    if g.rejected_thoughts:
        print("\nPruned (conflict-free, below threshold):")
        for n in g.rejected_thoughts:
            print(f"  - {n} ({topic_of.get(n, '?')})")

    # --- Per-topic: does the ground-truth claim outrank its RIVAL claim? ----
    # The head-to-head is between contending *claims* only: the ground-truth
    # claim vs. the claim(s) that contradict it. Corroborating observation
    # nodes (support-only, no conflict tag, not ground truth) are evidence, not
    # contenders, so they are excluded from "who ranks on top" — otherwise a
    # high-seed observation could masquerade as the topic's answer.
    print("\n" + "=" * 72)
    print("GROUND-TRUTH CHECK (ground-truth claim vs. rival claim, per topic)")
    print("=" * 72)

    def conf(n):
        return g.graph.nodes[n]["confidence"]

    def is_contender(n):
        meta = g.graph.nodes[n]["metadata"]
        return bool(meta.get("ground_truth") or meta.get("conflicts_with"))

    topics = {}
    for n in survivors:
        topics.setdefault(topic_of[n], []).append(n)

    hits = total = 0
    # Per-topic contender set captured here so the baseline ranks over the
    # exact same candidate claims as the engine (fair head-to-head).
    per_topic = {}
    for topic, ids in sorted(topics.items()):
        contested = any(g.graph.nodes[n]["metadata"].get("conflicts_with")
                        for n in ids)
        if not contested:
            continue
        truth_here = truth_ids & set(ids)
        if not truth_here:
            print(f"  {topic:<20} ground-truth claim did not survive; skipping")
            continue

        contenders = [n for n in ids if is_contender(n)]
        top = max(contenders, key=conf)
        truth_id = next(iter(truth_here))
        rivals = [n for n in contenders if n not in truth_here]
        rival_id = max(rivals, key=conf) if rivals else None

        total += 1
        correct = top in truth_here
        hits += correct
        mark = "PASS" if correct else "FAIL"
        rival_txt = (f"rival={rival_id} ({conf(rival_id):.3f})"
                     if rival_id else "rival=none survived")
        print(f"  [{mark}] {topic:<20} "
              f"ground_truth={truth_id} ({conf(truth_id):.3f})  vs  {rival_txt}"
              f"   -> top={top}")
        per_topic[topic] = {"contenders": contenders, "truth_id": truth_id,
                            "engine_pick": top}

    print("-" * 72)
    print(f"Ground-truth on top in {hits}/{total} contested topics.")

    # --- BASELINE: flat source_reliability, no graph reasoning -------------
    # "Trust the most reliable source": among the SAME contender claims, pick
    # the one with the highest raw seed reliability. No support propagation, no
    # contradiction handling, no confidence updates.
    print("\n" + "=" * 72)
    print("BASELINE GROUND-TRUTH CHECK (flat source_reliability, no propagation)")
    print("=" * 72)
    base_hits = 0
    for topic in sorted(per_topic):
        info = per_topic[topic]
        contenders, truth_id = info["contenders"], info["truth_id"]
        base_pick = max(contenders, key=lambda n: reliability_of[n])
        info["baseline_pick"] = base_pick
        rivals = [n for n in contenders if n != truth_id]
        rival_id = max(rivals, key=lambda n: reliability_of[n]) if rivals else None
        correct = base_pick == truth_id
        base_hits += correct
        mark = "PASS" if correct else "FAIL"
        rival_txt = (f"rival={rival_id} ({reliability_of[rival_id]:.2f})"
                     if rival_id else "rival=none")
        print(f"  [{mark}] {topic:<20} "
              f"ground_truth={truth_id} ({reliability_of[truth_id]:.2f})  vs  {rival_txt}"
              f"   -> pick={base_pick}")
    print("-" * 72)
    print(f"Baseline picks ground truth in {base_hits}/{total} contested topics.")

    # --- Side-by-side summary ---------------------------------------------
    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE: Baseline (flat reliability) vs Engine (graph propagation)")
    print("=" * 72)
    print(f"  {'Topic':<20} {'ground_truth':<22} "
          f"{'Baseline pick':<26} {'Engine pick':<26}")
    print("  " + "-" * 92)
    for topic in sorted(per_topic):
        info = per_topic[topic]
        truth_id = info["truth_id"]
        b, e = info["baseline_pick"], info["engine_pick"]
        b_txt = f"{b} ({'correct' if b == truth_id else 'WRONG'})"
        e_txt = f"{e} ({'correct' if e == truth_id else 'WRONG'})"
        print(f"  {topic:<20} {truth_id:<22} {b_txt:<26} {e_txt:<26}")
    print("  " + "-" * 92)
    print(f"  TOTALS:  Baseline {base_hits}/{total}   vs   Engine {hits}/{total}")

    return hits, total, base_hits


if __name__ == "__main__":
    main()
