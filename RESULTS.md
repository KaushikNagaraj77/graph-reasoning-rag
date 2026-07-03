# Experiment results

Full results for the contradiction-handling experiment. All numbers below are
produced by a single command:

```
python experiments/run_contradiction_experiment.py
```

## Corpus

36 claims across 8 contested topics, connected by 28 relationships
(`support` / `contradiction`). Each topic pits a **ground-truth** claim against
one or more rival claims. Five topics are "easy" (the ground-truth claim also
has the highest-reliability source); three are "hard" (the ground-truth claim
comes from a *lower*-reliability source and must be carried by corroborating
evidence):

| Topic | Type | Ground truth | Rival |
|-------|------|--------------|-------|
| dietary_fat | easy | fat_saturated_recent | fat_all_bad |
| vitamin_c_colds | easy | vitc_no_cure | vitc_prevents |
| hydration | easy | hydration_8cups_myth | hydration_8cups_rule |
| back_pain | easy | bedrest_harmful | bedrest_helpful |
| peptic_ulcers | easy | ulcer_bacteria | ulcer_stress |
| chronic_gastritis | **hard** | gastritis_infection | gastritis_acid |
| continental_drift | **hard** | drift_mobilist | drift_fixist |
| puerperal_fever | **hard** | childbed_handwash | childbed_miasma |

Source-reliability scores were assigned on source merit **blind to the answer**.
The correct-but-dismissed claims in the hard topics are genuinely scored low
(e.g. `drift_mobilist` 0.45, `gastritis_infection` ~0.50, `childbed_handwash`
~0.42); nothing was inflated to force a pass.

## Pipeline behavior

```
Loaded 36 claims across 8 topics, 28 relationships.
Pipeline: detected 11 contradiction pair(s); pruned 0 conflict-free
low-confidence claim(s).
Survivors: 36/36 claims kept.
```

All 36 claims survive. Contradictions are **kept and tagged** (`conflicts_with`),
not destroyed — the design choice that distinguishes this from the trading-origin
engine, which pruned conflicting nodes. Genuinely weak, unsupported sources still
sink to the confidence floor (blogs/forum posts bottom out at 0.150), so nothing
is kept indiscriminately.

## Head-to-head: engine vs. flat-reliability baseline

The **baseline** ranks each topic's competing claims purely by raw
source-reliability and takes the highest — no support propagation, no
contradiction handling. It ranks over the *same* per-topic contender set the
engine uses, so the comparison is apples-to-apples.

| Topic | Ground truth | Baseline pick | Engine pick |
|-------|--------------|---------------|-------------|
| back_pain | bedrest_harmful | ✅ correct | ✅ correct |
| chronic_gastritis | gastritis_infection | ❌ WRONG (gastritis_acid) | ✅ correct |
| continental_drift | drift_mobilist | ❌ WRONG (drift_fixist) | ✅ correct |
| dietary_fat | fat_saturated_recent | ✅ correct | ✅ correct |
| hydration | hydration_8cups_myth | ✅ correct | ✅ correct |
| peptic_ulcers | ulcer_bacteria | ✅ correct | ✅ correct |
| puerperal_fever | childbed_handwash | ❌ WRONG (childbed_miasma) | ❌ WRONG (childbed_miasma) |
| vitamin_c_colds | vitc_no_cure | ✅ correct | ✅ correct |
| **Total** | | **5 / 8** | **7 / 8** |

**Interpretation.** Engine and baseline agree on all 5 easy topics and both miss
`puerperal_fever` identically. The engine's entire advantage is on the two hard
topics (`chronic_gastritis`, `continental_drift`) where the correct claim's
source is *less* reliable than the wrong one, and support structure flips the
ranking. This is the intended effect, isolated: structure adds value exactly and
only where reliability alone is misleading.

## The hard cases in detail

| Topic | Ground-truth conf. | Rival conf. | Margin | Outcome |
|-------|--------------------|-------------|--------|---------|
| continental_drift | drift_mobilist 0.778 | drift_fixist 0.724 | +0.054 | engine flips ✅ |
| chronic_gastritis | gastritis_infection 0.738 | gastritis_acid 0.700 | +0.038 | engine flips ✅ |
| puerperal_fever | childbed_handwash 0.655 | childbed_miasma 0.692 | −0.037 | engine fails ❌ |

The `puerperal_fever` failure is explained by the mechanism, not by a bug. The
support boost is capped at +0.4; when the seed-reliability gap between a
correct-but-dismissed claim and an authoritative-but-wrong one is too large, the
capped boost cannot close it. `childbed_handwash` receives its full support boost
but still lands 0.037 below `childbed_miasma`. This is a **known, characterized
limit**: structure can overturn authority up to a bounded gap, beyond which
authority dominates.

## Verification

Every claim above was checked, not trusted:

- **Reporting logic** confirmed to be a fair per-topic head-to-head between rival
  claims (not a biased selection), and the baseline ranks over the identical
  contender set the engine uses.
- **Corpus honesty**: reliability scores assigned blind to the answer; an earlier
  iteration that tuned the corpus to pass was caught and reverted — the fix
  belonged in the engine (a contradiction-detection threshold inherited from the
  trading origin), not the data.
- **Arithmetic reproduced by hand**: the support-boost formula
  (`min(0.4, 0.15 × Σ supporter_reliability)`) and the resulting confidence
  values were recomputed by hand and matched the code's output for
  `continental_drift`.
- **Order-independence**: the result survives shuffling node order in the corpus,
  confirming the winning margins are structural, not artifacts of iteration order.
- **Baseline measured, not asserted**: the 5/8 baseline is computed in the same
  run as the 7/8 engine result.

## Scope and limits

- Small (8-topic), hand-authored corpus. This validates the *mechanism*, not
  generalization to real, LLM-extracted, noisy documents.
- Winning margins on hard cases are thin (+0.038, +0.054); the boost mechanism
  wins near the tipping point but cannot overcome large authority gaps
  (puerperal_fever).
- The underlying idea maps to established **truth-discovery** and
  **credibility-propagation** work (see README, *Relation to prior work*); this
  is a rigorously-validated reimplementation in a conflicting-claim setting, not
  a novel method.
