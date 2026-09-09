"""
Signal-reasoning demo: run the contradiction engine over one asset's sourced
claims and print an explained suggestion, side by side with the flat
"trust the most reliable source" baseline.

    python experiments/run_signal_demo.py
    python experiments/run_signal_demo.py --corpus path/to/corpus.json --mechanism additive
    python experiments/run_signal_demo.py --log data/decision_log.jsonl

The default corpus is a FICTIONAL asset; this is a mechanism demo, not advice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning.signals import (
    flat_baseline,
    format_suggestion,
    load_signal_corpus,
    log_suggestion,
    reason_signal,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "signal_corpus_sample.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--mechanism", choices=("logodds", "additive"),
                        default="logodds")
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--log", default=None,
                        help="Append the suggestion to this JSONL decision log.")
    args = parser.parse_args()

    asset, as_of, claims, relationships = load_signal_corpus(args.corpus)
    print(f"Loaded {len(claims)} claims and {len(relationships)} relationships "
          f"for {asset} (as of {as_of}).\n")

    s = reason_signal(asset, claims, relationships, mechanism=args.mechanism,
                      min_margin=args.min_margin, min_sources=args.min_sources,
                      as_of=as_of)

    print("=" * 78)
    print("ENGINE (graph propagation + contradiction handling)")
    print("=" * 78)
    print(format_suggestion(s))

    stance, cid = flat_baseline(claims)
    print("\n" + "=" * 78)
    print("BASELINE (stance of the single most reliable thesis source)")
    print("=" * 78)
    print(f"  {asset}: {stance or 'no thesis'}  [{cid or '-'}]")

    if args.log:
        path = log_suggestion(args.log, s)
        print(f"\nAppended suggestion to {path}")


if __name__ == "__main__":
    main()
