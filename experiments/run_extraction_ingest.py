"""
Stage 1a runner: extract claims from real documents, write proposed
relationships for human review, and STOP before the engine.

Flow:
  1. Load the document set (data/documents/metadata.json + text files).
  2. LLM-extract atomic claims (reliability comes from metadata).
  3. LLM-propose support/contradiction relationships.
  4. Write both to data/proposed_relationships.json for human review, then STOP.

Once a human has reviewed the proposals and written a verified file
(data/verified_relationships.json), re-run with --run-engine to build the
ReasoningGraph and print the same-style ground-truth check as the hand-authored
experiment.

    python experiments/run_extraction_ingest.py              # extract + propose, then stop
    python experiments/run_extraction_ingest.py --run-engine # requires verified file
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning.extract import (
    extract_claims,
    load_documents,
    load_extracted_into_graph,
    propose_relationships,
    write_proposed_relationships,
)

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "documents" / "metadata.json"
PROPOSED = ROOT / "data" / "proposed_relationships.json"
VERIFIED = ROOT / "data" / "verified_relationships.json"


def extract_and_propose():
    documents = load_documents(METADATA)
    print(f"Loaded {len(documents)} documents across "
          f"{len({d.topic for d in documents})} topics.\n")

    print("Extracting claims via LLM (reliability from metadata, not LLM-judged)...")
    claims = extract_claims(documents, verbose=True)
    print(f"\nExtracted {len(claims)} claims total.\n")

    print("=" * 72)
    print("EXTRACTED CLAIMS")
    print("=" * 72)
    for c in claims:
        gt = " [GROUND TRUTH]" if c["ground_truth"] else ""
        print(f"  {c['id']:<20} [{c['topic']}] r={c['source_reliability']:.2f}{gt}")
        print(f"      {c['claim_text']}")
        print(f"      source: {c['source']}")

    print("\nProposing relationships via LLM (for review, NOT fed to engine)...")
    relationships = propose_relationships(claims)

    print("\n" + "=" * 72)
    print("PROPOSED RELATIONSHIPS (REVIEW REQUIRED)")
    print("=" * 72)
    if not relationships:
        print("  (none proposed)")
    for r in relationships:
        print(f"  {r['from']}  --{r['type']}-->  {r['to']}")
        print(f"      rationale: {r['rationale']}")

    write_proposed_relationships(claims, relationships, PROPOSED)
    print("\n" + "-" * 72)
    print(f"Wrote {len(claims)} claims and {len(relationships)} proposed "
          f"relationships to:\n  {PROPOSED}")
    print("\nSTOPPED before the engine. Review the proposed relationships above.")
    print("To proceed: copy the reviewed file to")
    print(f"  {VERIFIED}")
    print("(editing/removing any wrong relationships), then re-run with --run-engine.")


def run_engine():
    if not VERIFIED.exists():
        print(f"No verified relationships file at {VERIFIED}.")
        print("Review the proposed relationships first, then create it. Aborting.")
        sys.exit(1)

    data = json.loads(VERIFIED.read_text())
    claims = data["claims"]
    relationships = data["relationships"]
    truth_ids = {c["id"] for c in claims if c.get("ground_truth")}
    topic_of = {c["id"]: c["topic"] for c in claims}

    g = load_extracted_into_graph(claims, relationships, name="extracted-medical")
    g.update_confidence_with_graph_structure()
    contradictions = g.detect_contradictions()
    g.resolve_contradictions(contradictions)
    g.prune_low_confidence(threshold=0.5)

    def conf(n):
        return g.graph.nodes[n]["confidence"]

    def is_contender(n):
        meta = g.graph.nodes[n]["metadata"]
        return bool(meta.get("ground_truth") or meta.get("conflicts_with"))

    survivors = list(g.graph.nodes)
    topics = {}
    for n in survivors:
        topics.setdefault(topic_of[n], []).append(n)

    print("=" * 72)
    print("GROUND-TRUTH CHECK ON EXTRACTED CORPUS (ground-truth vs rival claim)")
    print("=" * 72)
    hits = total = 0
    for topic, ids in sorted(topics.items()):
        if not any(g.graph.nodes[n]["metadata"].get("conflicts_with") for n in ids):
            continue
        truth_here = truth_ids & set(ids)
        if not truth_here:
            print(f"  {topic:<20} ground-truth claim did not survive; skipping")
            continue
        contenders = [n for n in ids if is_contender(n)]
        top = max(contenders, key=conf)
        truth_id = max(truth_here, key=conf)
        rivals = [n for n in contenders if n not in truth_here]
        rival_id = max(rivals, key=conf) if rivals else None
        total += 1
        hits += top in truth_here
        mark = "PASS" if top in truth_here else "FAIL"
        rival_txt = (f"rival={rival_id} ({conf(rival_id):.3f})"
                     if rival_id else "rival=none")
        print(f"  [{mark}] {topic:<20} "
              f"ground_truth={truth_id} ({conf(truth_id):.3f})  vs  {rival_txt}")
    print("-" * 72)
    print(f"Ground-truth on top in {hits}/{total} contested topics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-engine", action="store_true",
                        help="Build the graph from the VERIFIED relationships file.")
    args = parser.parse_args()
    if args.run_engine:
        run_engine()
    else:
        try:
            extract_and_propose()
        except RuntimeError as e:
            # Clear, single-line failure (e.g. missing credentials) instead of
            # a traceback into the SDK.
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(2)
