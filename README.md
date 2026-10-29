# ShotShift

**Measure how much a few-shot classifier's predictions move when its in-context examples are reordered — one call on any model returns per-query permutation cards and an aggregate report: stability score, flip rate, the accuracy spread a single ordering hides, and each example's leave-one-out influence.**

A few-shot prompt is usually evaluated once, in the order its author happened to write the examples. But language models are not indifferent to that order: permute the demonstrations and the predictions — and therefore the accuracy — move (Lu et al., *Fantastically Ordered Prompts*, ACL 2022; Zhao et al., *Calibrate Before Use*, ICML 2021). ShotShift turns that finding into an instrument: give it your examples, your queries, and any model behind a callable, and it runs the example set under many orders, then reports how stable the predictions actually were — per query and in aggregate — instead of letting one arbitrary ordering speak for the prompt.

## Install

ShotShift is not on PyPI yet. Install the GitHub version:

```
pip install "git+https://github.com/habib-analyst/shotshift.git"
```

For development:

```
git clone https://github.com/habib-analyst/shotshift.git
cd shotshift
python -m pip install -e ".[dev]"
python -m pytest
```

Requires Python 3.10+. The only runtime dependency is NumPy. Your model is a callable — a local Hugging Face model behind a scoring function, an API client that generates text, a hand-written rule: ShotShift renders the prompts, never imports your framework, and never sees your weights.

## Quickstart

```
from shotshift import Example, Query, probe

def score_fn(prompt: str) -> dict[str, float]:
    ...  # return a score (e.g. logprob) per label: {"positive": ..., "negative": ...}

examples = [
    Example("a glowing review", "positive"),
    Example("a dreadful review", "negative"),
    # ...
]
queries = [Query("some film", answer="positive"), ...]

report = probe(score_fn, examples, queries, max_orders=16, seed=0)
print(report.summary())
# ShotShift report: 48 queries, 16 order(s) of 8 example(s),
# stability ..., flip rate ..., fragile queries ...
#   canonical accuracy ...; mean over orders ...;
#   range [..., ...] (width ...); majority-vote accuracy ...
#   predicted-label share: positive ..., negative ...

card = report.cards[0]
print(card.canonical, card.modal, card.modal_share, card.n_flips)

report.to_dict()   # full artifact: every card, every prediction, full config
```

Two model protocols:

- **Score mode** (`mode="score"`): the callable returns a mapping of label to score — typically the logprob of the label continuation after the prompt. The prediction is the argmax label (ties break toward the earlier label in the label set).
- **Generate mode** (`mode="generate"`): the callable returns generated text, and a conservative parser reads the label out of it (explicit `Label: X` marker, then a leading label, then a bare label in a short response). Anything it cannot read is counted in the report's **unparsed rate**, never guessed.

Two companion probes:

```
from shotshift import probe_zero_shot, leave_one_out

zero = probe_zero_shot(score_fn, ["positive", "negative"], queries)  # no examples at all
loo = leave_one_out(score_fn, examples, queries, max_orders=4)       # each example's influence
```

## What the report measures

The experimental unit is the *prompt*: one example set, in one order, applied to every query. Predictions are labels, tracked per query across orders.

- **Stability score** — mean over queries of the modal share: how often the most-chosen label was chosen, across all orders. 1.0 means no prediction ever moved.
- **Flip rate** — share of non-canonical (order, query) runs whose prediction differs from the prediction at the supplied (identity) order.
- **Fragile queries** — share of queries whose prediction moves under at least one order, with per-query cards (canonical vs modal label, modal share, flip count) so the fragile items can be inspected, not just counted.
- **Accuracy block** (when gold labels are supplied) — accuracy at the supplied order, mean/min/max over orders, the **accuracy range** (max − min: the part of a single reported accuracy that is order luck), and **majority-vote accuracy** (scoring the modal label instead of the supplied-order prediction).
- **Predicted-label share** — the mean predicted-label distribution over orders; a prompt whose orders swing this distribution is exhibiting the majority/recency skew Zhao et al. describe, in measurable form.
- **Leave-one-out influence** — re-probing with each example removed and reporting the delta to mean accuracy and stability. A removal with a *positive* accuracy delta identifies an example that was hurting the set.

Order sets are exhaustive when the budget allows (`k! <= max_orders`) and a seeded sample of distinct permutations otherwise; order 0 is always the order you supplied. The report records its full config — mode, seed, the explicit order set, and the exact prompt templates — and `ShotReport.from_dict(report.to_dict())` round-trips it, so a run can be re-derived or challenged. A sampled run can never masquerade as an exhaustive one: the config says which it was.

## What ShotShift is not

- **Not its sibling, ChoiceShift.** ChoiceShift permutes the *answer options inside one multiple-choice question* and measures a benchmark item; ShotShift permutes the *demonstrations shared by a whole prompt* and measures the prompt, over free-form text labels, with leave-one-out example influence. Neither can express the other's experiment: ChoiceShift's questions have no demonstrations, and ShotShift has no options. They share a design discipline (permute one component, call one callable, report the movement), not a subject.
- **Not a prompt optimizer.** ShotShift does not search for a better order or a better example set, and it ships no calibration or correction of its own (Zhao et al.'s contextual calibration is a correction; this is a measurement). The majority-vote number is reported as what that scoring rule *would* have given, not as a fix.
- **Not a benchmark harness.** lm-eval, Inspect, and promptfoo run suites and report accuracy at the presented configuration. None of them reports an order-stability profile — per-query flip cards, an accuracy range over orders, leave-one-out example influence — as the product. ShotShift takes a prompt you already have and measures the one property they don't.
- **Not a judge of the model.** A prompt can be perfectly stable and reliably wrong — stability 1.0 at 50% accuracy on a balanced binary task is a consistent failure — and the report prints stability and accuracy side by side so neither can be quoted without the other.

## Measured demo (real model, real data)

`examples/sst2_order_demo.py` runs Qwen2.5-0.5B-Instruct (Apache-2.0, local CPU inference) on SST-2 sentiment classification. The example set is the first 4 negative and first 4 positive *training* rows of the published parquet, merged in file order; the queries are the first 48 *validation* rows (25 negative / 23 positive). 16 orders (the supplied order plus 15 seeded permutations of the 40,320 possible) × 48 queries, scored by the next-token logprob of ` positive` / ` negative`; plus a zero-shot pass and a leave-one-out pass at 4 orders. Full record: `examples/output/demo_summary.json`; every prediction: `examples/output/demo_report.json`.

| Metric | Measured |
|---|---|
| Stability score (mean modal share) | **0.987** |
| Flip rate (non-canonical runs that moved) | **0.033** |
| Queries whose prediction moves under ≥ 1 order | **10.4%** (5 of 48) |
| Accuracy at the supplied order | 0.938 |
| Mean accuracy over all 16 orders | 0.953 |
| Accuracy range over orders | **0.938 – 0.958 (width 0.021)** |
| Majority-vote-over-orders accuracy | 0.958 |
| Predicted-label share (true share: 0.479 positive) | positive 0.513 / negative 0.487 |
| Zero-shot accuracy, same queries, no examples | 0.958 |

Read plainly: **this prompt is stable, and ShotShift's demo is what a certified-stable prompt looks like.** The whole accuracy spread across 16 orders is one query wide (at n=48, one query is 2.1 points). The supplied order is one of the four lowest-scoring orders — and for one query (gold: negative) it is the *only* order of the 16 that predicts wrong: all 15 permutations predict negative, the as-written prompt alone predicts positive. That single card is the demo's sharpest output: not a benchmark claim, but a specific, inspectable defect in a specific prompt. The leave-one-out pass (4-order budget) names the set's load-bearing example: removing the first positive demonstration ("that loves its characters and communicates something rather beautiful about human nature") costs 3.6 points of mean accuracy and 4.7 points of stability; removing either of two other demonstrations *gains* half a point — within one-query noise at this size, and reported as such. Finally, the zero-shot comparison is reported, not buried: with no examples at all the model scores 0.958 on these queries, equal to the best few-shot orders — on this slice, the examples buy nothing, and reordering them costs at most one query. The large order swings in the literature were measured on GPT-3-era models; a small modern instruct model on an easy task is a different regime, and the instrument's job is to say which regime your prompt is in — with receipts either way.

Two scripted scorers in `examples/order_demo.py` demonstrate the mechanics with no model at all (a constant scorer scores stability 1.000; a first-example scorer scores 0.500 with flip rate 1.000). Those numbers describe the scripts, not any model, and the file says so.

## License

MIT. Issues and pull requests are welcome on GitHub.
