# graph-reasoning-rag

A contradiction-aware reasoning graph for conflicting-claim resolution, and a
rigorously verified experiment testing whether **corroborating evidence can
override source authority** when sources disagree.

This project implements ideas from **truth discovery** and **credibility
propagation** (see *Relation to prior work* below) in a conflicting-claim
GraphRAG setting. The emphasis is on end-to-end verification: every result here
was hand-checked, benchmarked against a proper baseline, and tested for
robustness rather than asserted.

---

## The question

Standard retrieval-augmented generation (RAG) retrieves the most *similar* text
chunks and passes them to a language model. It has no notion of which sources
are reliable, no detection of when retrieved passages *contradict* each other,
and no mechanism for corroborating evidence to outweigh a confident-but-wrong
source.

This project asks a narrower, testable question:

> When two claims conflict, can **support structure** (corroborating evidence)
> lift a well-evidenced but low-reliability claim above an authoritative-but-
> wrong one — something a flat "trust the most reliable source" baseline cannot
> do?

The motivating cases are real ones from the history of science, where an
initially-dismissed claim (low source authority) turned out correct because
evidence accumulated: continental drift, *H. pylori* as a cause of gastritis,
handwashing and puerperal fever.

## The engine

A directed graph where nodes are claims (each carrying a source and a
reliability score) and edges are typed relationships (`support`, `contradiction`).
Confidence is refined from graph topology:

- supporting edges boost a claim's confidence (diminishing, capped at +0.4)
- contradiction edges penalize it (capped at −0.4)
- contradictions are **kept and surfaced**, not destroyed — a claim in conflict
  is preserved and tagged, because a disagreement is information, not noise

The engine was extracted and generalized from a prior project (a since-retired
quantitative-trading system); the reasoning core is domain-agnostic and depends
only on `networkx` and `numpy`.

## The experiment

A hand-authored corpus of 8 topics. Five are "easy" (the reliable source is also
correct); three are "hard" (the correct claim comes from a *lower*-reliability
source and must be carried by corroborating evidence).

**Baseline:** flat source-reliability ranking — pick the most reliable source.
This is what a reliability-weighted retriever without structural reasoning does.

**Result:**

| System   | Score | Hard cases (correct source is low-reliability) |
|----------|-------|------------------------------------------------|
| Baseline | 5 / 8 | 0 / 3 — always picks the authoritative wrong claim |
| Engine   | 7 / 8 | 2 / 3 — support structure overturns authority   |

The engine and baseline are identical on the 5 easy topics. The entire advantage
is in the hard cases, where support structure lets a corroborated underdog win.
The one hard-case failure (puerperal fever) is explained by the mechanism: the
seed-reliability gap (~0.4) exceeds the boost cap, so structure *cannot* close
it. The advantage is real but bounded, and the winning margins on the two
successes are thin.

## Verification (the part I care most about)

Every claim in the result above was checked, not trusted:

- **Reporting logic** audited to confirm the ground-truth comparison is a fair
  head-to-head between rival claims, not a biased selection.
- **Corpus honesty**: source-reliability scores were assigned on source merit
  *blind to the answer*. The correct-but-dismissed claims are genuinely scored
  low (0.42–0.50); nothing was inflated to make the engine pass. An earlier
  iteration that *did* tune the corpus to pass was caught and reverted — the
  fix belonged in the engine (a contradiction-detection threshold that was an
  artifact of the trading origin), not the data.
- **Arithmetic reproduced by hand**: the confidence values were recomputed from
  the formula by hand and matched the code's output.
- **Order-independence**: the result survives shuffling node order in the corpus,
  confirming the winning margin is a structural fact, not an artifact of
  iteration order.
- **Baseline comparison**: the advantage is measured against a proper flat-
  reliability baseline, not asserted.

## Relation to prior work

The core idea here — jointly using source reliability and corroborating evidence
to resolve conflicting claims, with credibility propagating over support/
contradiction structure — sits within the **truth discovery** literature
(Yin, Han & Yu, *Truth Discovery with Multiple Conflicting Information Providers
on the Web*, 2008, and subsequent work), along with credibility-propagation
networks and evidence-aware fact verification. Confidence-aware variants that let
well-corroborated low-reliability sources outweigh authoritative ones are a known
line of work in that area.

This project is a from-scratch implementation of those principles in a
conflicting-claim GraphRAG setting. The contribution I'd point to is not the
underlying idea but the **rigor of the validation** — the mechanism is verified
by hand, benchmarked against a proper baseline, and shown to be robust rather
than assumed — and an explicit, honest account of how the approach relates to
existing methods.

## Limitations

- The corpus is small (8 topics) and hand-authored. This validates the
  *mechanism*, not generalization to real, LLM-extracted, noisy documents.
- The winning margins on hard cases are thin; the boost mechanism is weak enough
  that it wins near the tipping point but cannot overcome large authority gaps.
- No comparison yet against established learned methods from the truth-discovery
  literature — the natural next benchmark.

## Possible next steps

- Ingest **real documents** via LLM claim-extraction and test whether the effect
  survives extraction noise.
- Benchmark against established truth-discovery methods rather than only a naive
  baseline.
- Strengthen the evidence-accumulation mechanism so corroboration can decisively,
  not marginally, overcome authority.

## Structure

```
graph_reasoning/       the domain-agnostic reasoning engine (graph.py)
data/                  the hand-authored conflicting-claim corpus
experiments/           the contradiction experiment + baseline comparison
tests/                 standalone + contradiction-handling tests
docs/                  prior-project findings (the trading origin)
```

## Running it

```
pip install -r requirements.txt
python experiments/run_contradiction_experiment.py
```
