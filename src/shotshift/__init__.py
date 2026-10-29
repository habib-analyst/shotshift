"""ShotShift — few-shot example-order stability for LLM classifiers.

Permute the in-context examples of a few-shot prompt, run every order
over a query set through one model callable, and report how much the
predictions move: stability score, flip rate, fragile-query rate, the
accuracy spread a single ordering hides, and each example's
leave-one-out influence.
"""

from .core import Example, Query, QueryCard, ShotReport, build_report
from .parsing import parse_label
from .probe import leave_one_out, probe, probe_zero_shot
from .prompts import make_orders, render_prompt

__version__ = "0.1.0"

__all__ = [
    "Example",
    "Query",
    "QueryCard",
    "ShotReport",
    "build_report",
    "parse_label",
    "probe",
    "probe_zero_shot",
    "leave_one_out",
    "make_orders",
    "render_prompt",
    "__version__",
]
