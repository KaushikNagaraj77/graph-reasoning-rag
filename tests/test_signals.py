"""
Signal-reasoning tests (standalone script, same style as the other tests).

  1. Sample corpus: the engine SUGGESTS BUY where the flat baseline says
     bearish — the bull thesis has the weaker source but the stronger,
     more diverse evidence, and the evidence also refutes the bear thesis.
  2. Thin margin -> no_call.
  3. A leading thesis backed by a single source -> no_call.
  4. Well-backed bear thesis with no bullish thesis -> suggest_avoid.
  5. Validation rejects a bad stance.
  6. Decision log round-trips.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_reasoning.signals import (
    flat_baseline,
    load_signal_corpus,
    log_suggestion,
    reason_signal,
)

ROOT = Path(__file__).resolve().parents[1]

# --- 1. Sample corpus: engine flips the baseline ---------------------------
asset, as_of, claims, rels = load_signal_corpus(ROOT / "data" / "signal_corpus_sample.json")
base_stance, base_id = flat_baseline(claims)
assert base_stance == "bearish", f"baseline premise: most reliable thesis is bearish, got {base_stance}"

for mech in ("logodds", "additive"):
    s = reason_signal(asset, claims, rels, mechanism=mech, as_of=as_of)
    assert s.action == "suggest_buy", f"[{mech}] expected suggest_buy, got {s.action}: {s.reason}"
    assert s.bull_thesis == "bull_thesis", s.bull_thesis
    assert s.bear_thesis == "bear_thesis", s.bear_thesis
    assert s.margin >= 0.05, s.margin
    assert len(s.bull_sources) == 3, s.bull_sources
    assert "Retail forum post" not in s.bull_sources, "junk source must not count as corroboration"
    assert len(s.contradictions) == 2, s.contradictions  # mirrored pair reported once
    assert any(c["a"] == "ev_10q_growth" and c["b"] == "bear_thesis"
               for c in s.contradictions), "evidence->bear refutation must be surfaced"
    junk = next(r for r in s.claims if r["id"] == "forum_moon")
    assert junk["final"] < 0.3, f"junk source should stay low, got {junk['final']}"
    print(f"1. [{mech}] engine={s.action} (margin {s.margin:+.3f}) vs baseline={base_stance}")

# --- 2. Thin margin -> no_call -------------------------------------------------
thin_claims = [
    {"id": "b", "stance": "bullish", "claim_text": "up", "source": "S1", "source_reliability": 0.6},
    {"id": "r", "stance": "bearish", "claim_text": "down", "source": "S2", "source_reliability": 0.6},
    {"id": "e1", "stance": "neutral", "claim_text": "fact a", "source": "S3", "source_reliability": 0.8},
    {"id": "e2", "stance": "neutral", "claim_text": "fact b", "source": "S4", "source_reliability": 0.8},
]
thin_rels = [
    {"from": "b", "to": "r", "type": "contradiction"},
    {"from": "e1", "to": "b", "type": "support"},
    {"from": "e2", "to": "r", "type": "support"},
]
s = reason_signal("X", thin_claims, thin_rels)
assert s.action == "no_call" and "margin" in s.reason, (s.action, s.reason)
print(f"2. symmetric evidence -> {s.action} ({s.reason})")

# --- 3. Single-source winner -> no_call ---------------------------------------
single_claims = [
    {"id": "b", "stance": "bullish", "claim_text": "up", "source": "OneShop", "source_reliability": 0.9},
    {"id": "r", "stance": "bearish", "claim_text": "down", "source": "S2", "source_reliability": 0.4},
]
single_rels = [{"from": "b", "to": "r", "type": "contradiction"}]
s = reason_signal("X", single_claims, single_rels)
assert s.action == "no_call" and "distinct source" in s.reason, (s.action, s.reason)
# ...a junk corroborator does not change that...
single_claims.append({"id": "j", "stance": "neutral", "claim_text": "tip",
                      "source": "Anonymous tip", "source_reliability": 0.2})
single_rels.append({"from": "j", "to": "b", "type": "support"})
s1 = reason_signal("X", single_claims, single_rels)
assert s1.action == "no_call" and "distinct source" in s1.reason, (s1.action, s1.reason)
# ...but a credible second source makes the call.
single_claims.append({"id": "e", "stance": "neutral", "claim_text": "fact",
                      "source": "Filing", "source_reliability": 0.9})
single_rels.append({"from": "e", "to": "b", "type": "support"})
s2 = reason_signal("X", single_claims, single_rels)
assert s2.action == "suggest_buy", (s2.action, s2.reason)
print(f"3. single source -> {s.action}; junk corroborator -> {s1.action}; "
      f"credible corroborator -> {s2.action}")

# --- 4. Well-backed bear, no bull thesis -> suggest_avoid ---------------------
bear_claims = [
    {"id": "r", "stance": "bearish", "claim_text": "down", "source": "S1", "source_reliability": 0.6},
    {"id": "e1", "stance": "neutral", "claim_text": "fact a", "source": "S2", "source_reliability": 0.9},
    {"id": "e2", "stance": "neutral", "claim_text": "fact b", "source": "S3", "source_reliability": 0.9},
]
bear_rels = [{"from": "e1", "to": "r", "type": "support"},
             {"from": "e2", "to": "r", "type": "support"}]
s = reason_signal("X", bear_claims, bear_rels)
assert s.action == "suggest_avoid" and s.bull_thesis is None, (s.action, s.reason)
print(f"4. bear-only, corroborated -> {s.action}")

# --- 5. Validation ------------------------------------------------------------
try:
    reason_signal("X", [{"id": "z", "stance": "long", "claim_text": "t",
                         "source": "S", "source_reliability": 0.5}], [])
    raise AssertionError("bad stance must be rejected")
except ValueError as e:
    print(f"5. validation: {e}")

# --- 6. Decision log ----------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "log.jsonl"
    s = reason_signal(asset, claims, rels, as_of=as_of)
    log_suggestion(path, s)
    log_suggestion(path, s, include_claims=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["action"] == "suggest_buy"
    assert "claims" not in rows[0] and "claims" in rows[1]
    assert rows[0]["as_of"] == as_of and "logged_at" in rows[0]
print("6. decision log round-trips")

print("\nSIGNAL REASONING TEST: PASS")
