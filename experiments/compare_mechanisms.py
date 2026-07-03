"""
Compare confidence-update mechanisms against the full test suite.

This is a COMPARISON, not a commitment. The engine's default mechanism is left
untouched; each alternative is implemented here as a selectable, self-contained
confidence pass that operates on a freshly-seeded graph. All four are scored with
the SAME rival-identification and PASS/FAIL logic so the comparison is apples to
apples.

Datasets (the full suite):
  1. data/sample_corpus.json                     — 5 easy + 3 hand-authored hard
                                                    (incl. puerperal_fever)
  2. data/proposed_relationships.json            — 2 easy real-extracted
                                                    (continental_drift, peptic_ulcers)
  3. data/verified_relationships_hardcase.json   — 1915 real-extracted hard case
                                                    (the current failure / target)

A topic PASSES if its ground-truth claim ends with strictly higher confidence
than the highest-confidence outdated claim it actually CONTRADICTS (rival found
via contradiction edges, per the corrected check — evidence/support nodes are
never treated as rivals).

Mechanisms (reasonable, STATED parameters; constants are not tuned to force a
pass):

  current  (baseline, matches the engine)
      final = clamp( s
                     + min(0.4, 0.15 * Σ_sup mass)
                     - min(0.4, 0.20 * Σ_con mass) )
      single pass; mass = edge_conf * predecessor_confidence.

  A  CONVERGENT EVIDENCE — independent supporters compound (reduce doubt)
      doubt = 1 - s
      s_sup = 1 - (1 - s) * Π_i (1 - ALPHA * mass_i)     # supporters compound
      final = s_sup * Π_j (1 - BETA * mass_j)            # contradictors compound
      ALPHA = 0.5, BETA = 0.35. More independent lines -> more lift, no
      diminishing-returns cap. Single pass over seed masses.

  B  DISPUTED-AUTHORITY DISCOUNT — penalty scales with the attacker's CURRENT
     confidence, not its seed. Iterated to a fixed point.
      boost_A  = min(0.4, 0.15 * Σ_sup mass)             # same additive boost
      pen_A    = Σ_B BETA_B * edge_conf(B->A) * conf_B   # conf_B is CURRENT
      final_A  = clamp( s_A + boost_A - pen_A )
      BETA_B = 0.30. Iterated with damping 0.5 until max|Δ| < 1e-4 (cap 100
      iters); a weakened attacker's attack weakens too.

  C  BAYESIAN LOG-ODDS — seed is a prior; evidence updates it in log-odds.
      L0    = logit(clamp(s, 0.02, 0.98))
      L     = L0 + W * Σ_sup mass - W * Σ_con mass
      final = sigmoid(L)
      W = 1.5. Strong evidence can move a low prior a long way; symmetric for
      support and contradiction. Single pass over seed masses.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning import RelationType, ReasoningGraph

ROOT = Path(__file__).resolve().parents[1]

SUPPORT_TYPES = {RelationType.SUPPORT.value, RelationType.CONFLUENCE.value,
                 RelationType.VALID_INFERENCE.value, RelationType.DERIVATION.value}
CONTRA_TYPES = {RelationType.CONTRADICTION.value, RelationType.FALLACY.value}

MECHANISMS = ("current", "A", "B", "C")


# ---------------------------------------------------------------------------
# Corpus loading — normalize every dataset to (claims, relationships)
# ---------------------------------------------------------------------------

def load_dataset(path):
    data = json.loads(Path(path).read_text())
    return data["claims"], data["relationships"]


def build_graph(claims, relationships):
    """Seed a graph exactly as the real loaders do (confidence = reliability)."""
    g = ReasoningGraph("compare")
    for c in claims:
        g.add_thought(c["id"], c["claim_text"],
                      confidence=c["source_reliability"],
                      metadata={"source": c.get("source"),
                                "source_reliability": c["source_reliability"],
                                "topic": c["topic"],
                                "ground_truth": c.get("ground_truth", False)})
    rel_map = {"support": RelationType.SUPPORT,
               "contradiction": RelationType.CONTRADICTION}
    for r in relationships:
        rt = rel_map[r["type"]]
        edge_conf = g.graph.nodes[r["from"]]["metadata"]["source_reliability"]
        g.add_relation(r["from"], r["to"], rt, confidence=edge_conf)
    return g


# ---------------------------------------------------------------------------
# Edge-mass helpers (mass = edge_conf * predecessor CURRENT confidence)
# ---------------------------------------------------------------------------

def incoming(g, node, conf):
    """Return (support_masses, contra_masses) using confidences in `conf`."""
    sup, con = [], []
    for pred in g.graph.predecessors(node):
        edge = g.graph[pred][node]
        m = edge.get("confidence", 0.5) * conf[pred]
        if edge.get("type") in SUPPORT_TYPES:
            sup.append((pred, edge.get("confidence", 0.5), m))
        elif edge.get("type") in CONTRA_TYPES:
            con.append((pred, edge.get("confidence", 0.5), m))
    return sup, con


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# The four mechanisms — each returns {node: final_confidence}
# ---------------------------------------------------------------------------

def mech_current(g, seed):
    # Faithful reproduction of the engine: a SINGLE pass in node order that
    # MUTATES confidences in place, so a node visited later sees its
    # predecessors' already-updated values (matches
    # ReasoningGraph.update_confidence_with_graph_structure exactly).
    conf = dict(seed)
    for n in g.graph.nodes:
        sup, con = incoming(g, n, conf)
        s = seed[n]
        if sup:
            s += min(0.4, 0.15 * sum(m for _, _, m in sup))
        if con:
            s -= min(0.4, 0.20 * sum(m for _, _, m in con))
        conf[n] = clamp(s)
    return conf


def mech_A(g, seed, ALPHA=0.5, BETA=0.35):
    out = {}
    for n in g.graph.nodes:
        sup, con = incoming(g, n, seed)
        s = seed[n]
        doubt = 1.0 - s
        for _, _, m in sup:
            doubt *= (1.0 - ALPHA * clamp(m))
        s_sup = 1.0 - doubt
        conf = s_sup
        for _, _, m in con:
            conf *= (1.0 - BETA * clamp(m))
        out[n] = clamp(conf)
    return out


def mech_B(g, seed, BETA_B=0.30, damping=0.5, max_iter=100, tol=1e-4):
    conf = dict(seed)
    iters = 0
    for _ in range(max_iter):
        iters += 1
        new = {}
        for n in g.graph.nodes:
            sup, con = incoming(g, n, conf)  # penalty uses CURRENT conf
            s = seed[n]
            if sup:
                s += min(0.4, 0.15 * sum(m for _, _, m in sup))
            # disputed-authority discount: attacker's CURRENT confidence
            pen = sum(BETA_B * edge_conf * conf[pred] for pred, edge_conf, _ in con)
            new[n] = clamp(s - pen)
        damped = {n: conf[n] + damping * (new[n] - conf[n]) for n in new}
        delta = max(abs(damped[n] - conf[n]) for n in damped) if damped else 0.0
        conf = damped
        if delta < tol:
            break
    conf["_iters"] = iters  # carried out-of-band for reporting
    return conf


def _logit(p):
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def mech_C(g, seed, W=1.5):
    out = {}
    for n in g.graph.nodes:
        sup, con = incoming(g, n, seed)
        L = _logit(clamp(seed[n], 0.02, 0.98))
        L += W * sum(m for _, _, m in sup)
        L -= W * sum(m for _, _, m in con)
        out[n] = clamp(_sigmoid(L))
    return out


def run_mechanism(mech, g, seed):
    if mech == "current":
        return mech_current(g, seed), None
    if mech == "A":
        return mech_A(g, seed), None
    if mech == "B":
        conf = mech_B(g, seed)
        iters = conf.pop("_iters", None)
        return conf, iters
    if mech == "C":
        return mech_C(g, seed), None
    raise ValueError(mech)


# ---------------------------------------------------------------------------
# Scoring — GT vs its TRUE rival (contradiction-edge rival, corrected check)
# ---------------------------------------------------------------------------

def score(claims, relationships, final):
    """Return {topic: (passed, gt_id, gt_conf, rival_id, rival_conf)} per contested topic."""
    gt_of = {c["id"]: bool(c.get("ground_truth")) for c in claims}
    topic_of = {c["id"]: c["topic"] for c in claims}
    truth_ids = {c["id"] for c in claims if c.get("ground_truth")}

    # GT -> set of outdated (gt=false) claims it contradicts, via either direction
    gt_rivals = {t: set() for t in truth_ids}
    for r in relationships:
        if r["type"] != "contradiction":
            continue
        a, b = r["from"], r["to"]
        for x, y in ((a, b), (b, a)):
            if x in truth_ids and gt_of.get(y) is False:
                gt_rivals[x].add(y)

    topics = {}
    for c in claims:
        topics.setdefault(c["topic"], []).append(c["id"])

    results = {}
    for topic, ids in topics.items():
        truth_here = [t for t in truth_ids if t in ids]
        if not truth_here:
            continue
        # Representative GT claim: one that actually contradicts an outdated
        # claim; among those, the highest final confidence.
        contesting = [t for t in truth_here if gt_rivals.get(t)]
        if not contesting:
            continue
        gt = max(contesting, key=lambda n: final[n])
        rivals = list(gt_rivals[gt])
        if not rivals:
            continue
        rival = max(rivals, key=lambda n: final[n])
        passed = final[gt] > final[rival]
        results[topic] = (passed, gt, final[gt], rival, final[rival])
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DATASETS = [
    ("sample_corpus", ROOT / "data" / "sample_corpus.json"),
    ("real_extracted_easy", ROOT / "data" / "proposed_relationships.json"),
    # CLEAN re-extraction of the 1915 hard case (14 claims, evidence
    # concentrated on the core "continents drifted" claim). The stale
    # 35-claim fragmented version lives in verified_relationships_hardcase.json.
    ("real_extracted_1915", ROOT / "data" / "proposed_relationships_hardcase.json"),
]

# Topics we expect to be "already passing" under the baseline (for regression
# flagging) are determined empirically from the current mechanism, not assumed.


def evaluate_all():
    # topic -> {mech: (passed, gt, gtc, rival, rc)}
    per_topic = {}
    topic_dataset = {}
    b_iters = {}

    for ds_name, path in DATASETS:
        claims, rels = load_dataset(path)
        seed = {c["id"]: c["source_reliability"] for c in claims}
        g = build_graph(claims, rels)
        for mech in MECHANISMS:
            final, iters = run_mechanism(mech, g, seed)
            if mech == "B" and iters is not None:
                b_iters[ds_name] = iters
            res = score(claims, rels, final)
            for topic, tup in res.items():
                key = f"{ds_name}:{topic}"
                per_topic.setdefault(key, {})[mech] = tup
                topic_dataset[key] = ds_name

    return per_topic, topic_dataset, b_iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=MECHANISMS + ("all",),
                        default="all",
                        help="Run one mechanism, or 'all' for the comparison.")
    args = parser.parse_args()

    per_topic, topic_dataset, b_iters = evaluate_all()
    topics = sorted(per_topic)
    mechs = MECHANISMS if args.mechanism == "all" else (args.mechanism,)

    # Baseline pass/fail for regression detection.
    baseline = {t: per_topic[t]["current"][0] for t in topics}

    # --- Per-mechanism detail tables --------------------------------------
    for mech in mechs:
        print("=" * 78)
        label = "current (baseline)" if mech == "current" else f"mechanism {mech}"
        print(f"MECHANISM: {label}")
        print("=" * 78)
        print(f"  {'topic':<34} {'GT conf':>8} {'rival':>8}  result")
        print("  " + "-" * 62)
        passed_n = 0
        for t in topics:
            p, gt, gtc, rv, rc = per_topic[t][mech]
            passed_n += p
            mark = "PASS" if p else "FAIL"
            regress = ""
            if mech != "current" and baseline[t] and not p:
                regress = "  <-- REGRESSION"
            print(f"  {t:<34} {gtc:>8.3f} {rc:>8.3f}  [{mark}]{regress}")
        print("  " + "-" * 62)
        extra = ""
        if mech == "B" and b_iters:
            iters_txt = ", ".join(f"{ds}={it}" for ds, it in b_iters.items())
            extra = f"   (B iterated to convergence: {iters_txt})"
        print(f"  SCORE: {passed_n}/{len(topics)} topics pass{extra}")
        print()

    if args.mechanism != "all":
        return

    # --- Side-by-side summary ---------------------------------------------
    print("=" * 78)
    print("SIDE-BY-SIDE (rows = topics, cols = mechanisms; * = regression vs baseline)")
    print("=" * 78)
    header = f"  {'topic':<34} " + " ".join(f"{m:>8}" for m in MECHANISMS)
    print(header)
    print("  " + "-" * (34 + 9 * len(MECHANISMS)))
    scores = {m: 0 for m in MECHANISMS}
    for t in topics:
        cells = []
        for m in MECHANISMS:
            p = per_topic[t][m][0]
            scores[m] += p
            tag = "P" if p else "F"
            if m != "current" and baseline[t] and not p:
                tag = "F*"
            cells.append(f"{tag:>8}")
        print(f"  {t:<34} " + " ".join(cells))
    print("  " + "-" * (34 + 9 * len(MECHANISMS)))
    print(f"  {'SCORE (passed/total)':<34} "
          + " ".join(f"{str(scores[m])+'/'+str(len(topics)):>8}" for m in MECHANISMS))

    # --- Targeted findings -------------------------------------------------
    def find(substr):
        for t in topics:
            if t.endswith(substr):
                return t
        return None

    t1915 = find(":continental_drift_1915")
    tpf = find(":puerperal_fever")

    print("\n" + "=" * 78)
    print("TARGETED FINDINGS")
    print("=" * 78)
    for m in MECHANISMS:
        if m == "current":
            continue
        flips_1915 = (t1915 and not per_topic[t1915]["current"][0]
                      and per_topic[t1915][m][0])
        flips_pf = (tpf and not per_topic[tpf]["current"][0]
                    and per_topic[tpf][m][0])
        regressions = [t for t in topics
                       if baseline[t] and not per_topic[t][m][0]]
        print(f"\nMechanism {m}:")
        print(f"  flips 1915 hard case (TARGET): {'YES' if flips_1915 else 'no'}")
        print(f"  flips puerperal_fever (BONUS): {'YES' if flips_pf else 'no'}")
        if regressions:
            print(f"  REGRESSIONS ({len(regressions)}) — DISQUALIFYING: "
                  + ", ".join(regressions))
        else:
            print("  regressions: none")

    print("\n" + "-" * 78)
    print("Legend: P=pass  F=fail  F*=fail that REGRESSED a baseline-passing topic.")
    print("A mechanism with any F* is disqualified regardless of what it flips.")


if __name__ == "__main__":
    main()
