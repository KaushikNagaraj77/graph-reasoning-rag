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

## Testing on real documents — and where it broke

The hand-authored result validated the mechanism on a clean, controlled graph.
The harder question was whether it survives real documents, so I built an
extraction pipeline (LLM proposes claims + relationships from real text → human
verifies → engine runs) and tested it on a deliberately hard case: a
dismissed-at-the-time (low-reliability) continental-drift claim against the
authoritative fixist consensus.

It failed at first — and diagnosing *why*, one variable at a time, was the most
useful part of the project. The reasoning engine was never the bottleneck; two
**LLM-extraction** flaws were:

- **Claim fragmentation** — the LLM split one position into many near-duplicate
  claims and scattered the supporting evidence across them, so the evidence never
  concentrated on the claim under comparison.
- **Missing refutation edges** — the LLM drew *support* edges but not the
  *contradiction* edges where evidence refutes a rival, so the
  authoritative-but-wrong claim self-reinforced unopposed. (Evidence is bipolar:
  corroborating a claim also disconfirms its rivals.)

Correcting both at the extraction step — consolidating duplicate claims,
attaching evidence to the core claim, and proposing *genuine* refutation edges —
let the correct claim overcome even a large authority gap, without regressing any
previously-passing case. Full write-up: [`docs/WRITEUP.md`](docs/WRITEUP.md).

The takeaway: **the quality of graph reasoning is hostage to the quality of graph
construction.** Where the graph is extracted from unstructured text, extraction
quality — not the propagation rule — is what determines whether reasoning works.
Graph reasoning is strongest when the structure is a trustworthy given (curated
databases, ontologies, recorded relationships) rather than extracted unreliably
from prose.

## Honest scope

- All results are on small (8–14 topic) hand-built corpora. This validates the
  mechanism and the diagnosis, not generalization or statistical significance.
- The core idea is **not novel** — it maps onto established truth-discovery work.
- Confidence-propagation engines of this kind are vulnerable to collusion (many
  low-reliability sources corroborating one false claim); guarding against that
  without collapsing to a flat reliability rule is unresolved.
- Not yet benchmarked against established truth-discovery datasets — the natural
  next validation.

## Signal reasoning (toward research agents)

`graph_reasoning/signals.py` points the same engine back at its original
domain, but as a **reasoning and suggestion layer, not an execution layer**.
Given sourced, stance-tagged claims about one asset — `bullish`/`bearish`
*thesis* claims and `neutral` *evidence* claims — plus verified
support/contradiction edges, `reason_signal` returns an explained
`Suggestion`: `suggest_buy`, `suggest_avoid`, or `no_call`, with both theses,
their scores, the sources behind each, and every contradiction found.

Two abstention guards are built in, because the prior project's lesson was
that unguarded signals do not survive contact with costs:

- **thin margin → `no_call`**: the evidence does not separate the theses;
- **single credible source → `no_call`**: a leading thesis backed by one
  source (junk sources don't count) is not corroborated. This is the simplest
  form of the collusion guard listed under *Honest scope*.

Thesis-vs-thesis contradictions are mirrored so both sides are penalized;
evidence-vs-thesis refutations stay directed. `log_suggestion` appends each
call with its `as_of` date to a JSONL decision log so hit-rate can be scored
forward against realized returns net of costs — a suggestion that is never
scored is just an opinion.

```
python experiments/run_signal_demo.py            # fictional sample asset
python tests/test_signals.py
```

On the sample corpus the flat baseline follows the more reliable (bearish)
newsletter; the engine suggests buy because the weaker sell-side thesis is
backed by three primary sources, one of which directly refutes the bear
thesis. `extract_stance_claims` in `extract.py` produces this input from real
documents (reliability still from metadata; relationships still human-gated),
but it has not yet been run against the live API here. What is **not** built:
any measurement that these suggestions predict returns. That forward test is
the next step, and until it exists this layer is a reasoning tool, not a
signal.

## Structure

```
graph_reasoning/       the domain-agnostic reasoning engine (graph.py) + extraction
                       + signals.py (asset thesis reasoning -> explained suggestion)
data/                  the hand-authored corpus + real-document sets + signal sample
experiments/           the contradiction experiment, baseline, and mechanism comparison
tests/                 standalone + contradiction-handling tests
docs/                  WRITEUP.md (the extraction-flaws write-up) + prior-project findings
```

## Running it

```
pip install -r requirements.txt
python experiments/run_contradiction_experiment.py
```
