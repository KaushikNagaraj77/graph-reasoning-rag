# When LLM Extraction Flaws Break Graph-Based Reasoning

A build-and-diagnose writeup of a contradiction-aware reasoning graph — what it
does, where it broke on real documents, and what the failures taught me about
the messy interface between LLM extraction and structured graph logic.

This is an engineering writeup, not a research paper. The core idea is not novel
(it maps onto the established *truth discovery* literature — see *Relation to
prior work*). The value here is the process: build it, verify every claim by
hand, benchmark honestly, and diagnose failures one variable at a time.

---

## The system

A directed graph where nodes are claims — each carrying a source and a
reliability score — and edges are typed relationships (`support`,
`contradiction`). Confidence propagates through the topology so that corroborating
evidence can lift a low-reliability-but-correct claim above an
authoritative-but-wrong one. Contradictions are kept and surfaced, not destroyed:
a claim in conflict is preserved and tagged, because a disagreement is
information.

The reasoning core was extracted from a prior quantitative-trading system that I
tested rigorously and found had no tradeable edge. The engine itself was
domain-general, so I stripped the trading specifics and repointed it at
conflicting-claim resolution.

## The controlled result

On a hand-authored 8-topic corpus, the engine beat a flat source-reliability
baseline **7/8 vs 5/8** — but the advantage was entirely on the "hard" cases,
where the correct claim came from a *lower*-reliability source and had to be
carried by corroborating evidence. On the "easy" cases (correct claim = most
reliable source), the engine and the baseline tied: structure added nothing a
lookup wouldn't.

That scoping is the honest headline: **structural propagation earns its keep only
when source reliability and truth diverge.** Everywhere else, a reliability sort
does just as well.

Every number in that result was checked, not trusted: the boost arithmetic was
reproduced by hand, the winning margins were shown to survive node-order
perturbation, and an earlier attempt to tune the corpus to pass was caught and
reverted (the fix belonged in the engine, not the data).

## Where it broke: real documents

The controlled corpus was hand-authored — one clean claim per position, evidence
pointing exactly where it should. To test whether the mechanism survived reality,
I built an extraction pipeline: an LLM reads real documents and proposes claims
and relationships, a human verifies them, then the engine runs.

On *easy* real-document topics it passed — but only because the correct claims
also had high source reliability, so a flat baseline would have passed too. That
validated the pipeline plumbing, not the mechanism.

The interesting failures came from a deliberately *hard* real-document case: a
1915-era "continental drift is real" claim (dismissed at the time, so genuinely
low reliability) against the authoritative "continents are fixed" consensus. Two
distinct extraction flaws broke the graph's reasoning — and neither was a flaw in
the reasoning itself.

### Flaw 1 — Claim fragmentation dilutes evidence

The LLM split one position ("the continents drifted") into nine near-duplicate
and closely-related claims, then scattered the physical evidence (coastline fit,
matching fossils, matching strata) across them. The specific claim the
evaluation compared received almost none of the evidence directly — it was routed
mostly to a *different* fragment of the same position.

The evidence existed in the graph. It just never reached the node under
comparison. No confidence formula can amplify support that isn't there.

This is a predictable consequence of LLM knowledge-graph construction: models
extract literal variations of a claim rather than grounding them to one unified
concept, because they lack robust coreference resolution out of the box.
Consolidating near-duplicate claims at extraction time (and attaching evidence to
the *core* claim of a position) fixed the dilution.

### Flaw 2 — Missing refutation edges let authority self-reinforce

Even with clean extraction, the hard case still failed — and the reason was
subtle. The evidence *supported* the correct claim but never *contradicted* the
opposing one. So the authoritative-but-wrong claim, reinforced by its own peer
claims and attacked by nothing, sailed to near-certainty while the correct claim
climbed but couldn't catch up.

The missing insight: **evidence is bipolar.** Corroborating evidence for a claim
simultaneously *disconfirms its rivals* — identical fossils across an ocean don't
just support drift, they refute "the continents never moved." But the extraction
drew only the support edges, never the refutation edges. The graph modeled
evidence as unipolar when it is bipolar.

This, too, is a predictable LLM-extraction limitation: processing documents in a
shallow pass, a model rarely infers the latent contradiction between a claim in
one document and an implication in another unless explicitly prompted to
cross-examine them.

Adding the *genuine* refutation edges — verified as real logical refutations, not
manufactured to force a result — let the evidence finally attack the authority.
Under a convergent-evidence propagation rule, the correct claim overtook the
authoritative-but-wrong one, and did so without regressing any previously-passing
case.

## The takeaway

The reasoning engine was never really the bottleneck. In every hard failure, the
problem lived at the **interface between LLM extraction and graph structure** —
fragmentation diluting evidence, and missing refutation edges leaving authority
unopposed. The graph-propagation "physics" was only as good as the graph the
extraction handed it.

That's the practical lesson for anyone building knowledge graphs from documents
with an LLM: the extraction quality — coreference/consolidation, and whether the
model captures what evidence argues *against*, not just what it supports —
determines whether downstream graph reasoning works at all. The clever
propagation rule is downstream of, and hostage to, the messy extraction step.

## Relation to prior work

The core idea — jointly using source reliability and corroborating evidence to
resolve conflicting claims, with credibility propagating over support/
contradiction structure — sits within the **truth discovery** literature (Yin,
Han & Yu, 2008 onward), and relates to abstract argumentation frameworks (Dung,
1995) and trust/credibility propagation. I arrived at the approach independently
and then found it was an established field; this is a from-scratch
reimplementation of those principles, valued for the engineering and the honest
diagnosis rather than for novelty.

## Honest limitations

- All results are on small (8–14 topic) hand-built corpora. This validates the
  mechanism and the diagnosis, not generalization or statistical significance.
- Confidence-propagation engines of this kind are vulnerable to collusion: a
  cluster of low-reliability sources corroborating the same false claim could
  aggregate undue weight. Guarding against that without collapsing back to a flat
  reliability rule is an unresolved tension.
- The mechanism choice rests on a narrow margin (one topic) on a small suite —
  suggestive, not settled.
- Real-world contradictions are rarely clean A-vs-¬A; framing, nuance, and
  context-shift make contradiction detection much harder in the open domain than
  in a curated corpus.

## What I'd do next (if continuing)

Not tune this further — the returns are diminishing on a small corpus. The
genuinely open threads are: validating the extraction-quality findings at scale
on real corpora, benchmarking against established truth-discovery datasets, and —
the direction I'm actually pursuing — graph/GNN foundations, where learned
representations might handle the extraction noise that rigid, hand-coded
propagation cannot.
