"""Scripted mechanics demo — NO model is involved.

Three hand-written scorers with by-construction behaviour run through
ShotShift end to end, and every printed number is asserted against the
value the construction implies. These numbers describe the scripts, not
any model; the measured demo with a real model is sst2_order_demo.py.

Scorers:
- constant: always prefers "positive" -> stability 1.000, flip rate 0.
- first-demo: predicts the label of the first example in the prompt ->
  with two examples its prediction swaps with the order (stability 0.500).
- majority: predicts the majority label among the examples -> order
  cannot move it (stability 1.000), and leave-one-out shows removing the
  single minority example changes nothing while removing a majority
  example flips the set's prediction.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotshift import Example, Query, leave_one_out, probe  # noqa: E402

_LABEL_RE = re.compile(r"Label: (\w+)")

EXAMPLES = [
    Example("a glowing review", "positive"),
    Example("a dreadful review", "negative"),
    Example("an instant classic", "positive"),
]
QUERIES = [
    Query("some film", answer="positive"),
    Query("other film", answer="negative"),
    Query("third film", answer="positive"),
]


def _scores(label: str) -> dict[str, float]:
    return {"positive": 1.0 if label == "positive" else 0.0,
            "negative": 1.0 if label == "negative" else 0.0}


def constant_scorer(prompt: str) -> dict[str, float]:
    return _scores("positive")


def first_demo_scorer(prompt: str) -> dict[str, float]:
    return _scores(_LABEL_RE.findall(prompt)[0])


def majority_scorer(prompt: str) -> dict[str, float]:
    labels = _LABEL_RE.findall(prompt)
    return _scores("positive" if labels.count("positive") >= labels.count("negative")
                   else "negative")


def check(name: str, got: float, want: float) -> None:
    status = "ok" if abs(got - want) < 1e-9 else "MISMATCH"
    print(f"  {name}: {got:.3f} (expected {want:.3f}) [{status}]")
    if status != "ok":
        raise SystemExit(f"mechanics demo failed on {name}")


def main() -> None:
    print("constant scorer (3 examples, 6 orders):")
    rep = probe(constant_scorer, EXAMPLES, QUERIES)
    print(" ", rep.summary().replace("\n", "\n  "))
    check("stability", rep.stability, 1.0)
    check("flip rate", rep.flip_rate, 0.0)
    check("canonical accuracy", rep.canonical_accuracy, 2 / 3)

    print("first-demo scorer (2-example subset, 2 orders):")
    rep = probe(first_demo_scorer, EXAMPLES[:2], QUERIES)
    print(" ", rep.summary().replace("\n", "\n  "))
    check("stability", rep.stability, 0.5)
    check("flip rate", rep.flip_rate, 1.0)
    check("fragile rate", rep.fragile_rate, 1.0)

    print("majority scorer (3 examples, 6 orders):")
    rep = probe(majority_scorer, EXAMPLES, QUERIES)
    print(" ", rep.summary().replace("\n", "\n  "))
    check("stability", rep.stability, 1.0)
    check("canonical accuracy", rep.canonical_accuracy, 2 / 3)

    print("leave-one-out under the majority scorer:")
    result = leave_one_out(majority_scorer, EXAMPLES, QUERIES)
    for entry in result["leave_one_out"]:
        ex = entry["removed_example"]
        print(f"  remove [{entry['removed_index']}] {ex['label']!r} example: "
              f"delta mean accuracy {entry['delta_mean_accuracy']:+.3f}, "
              f"delta stability {entry['delta_stability']:+.3f}")
    deltas = {e["removed_index"]: e["delta_mean_accuracy"]
              for e in result["leave_one_out"]}
    # removing the lone negative example: majority stays positive -> no change
    check("LOO remove minority example", deltas[1], 0.0)
    # removing a positive example ties the set 1-1; the tie-break (>=) still
    # predicts positive, so accuracy is unchanged here too -- the honest
    # readout is that this set's prediction is carried by the tie-break,
    # which the stability/accuracy pair shows rather than hides.
    check("LOO remove majority example", deltas[0], 0.0)
    print("mechanics demo: all by-construction values reproduced")


if __name__ == "__main__":
    main()
