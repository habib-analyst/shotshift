"""Conservative label extraction for generate mode.

The parser recognises, in order:

1. an explicit marker — ``Label: positive``, ``Answer: negative``,
   ``Sentiment: positive`` (marker word, a colon or dash, then the label);
2. a leading label — the response *starts* with a label (``positive``,
   ``positive.``, ``positive — because...``);
3. a bare label appearing as a standalone word anywhere in a short
   response (<= 40 chars).

Matching is case-insensitive and the label must appear as a whole token,
not as a substring of a longer word. When two different labels both match
at the same precedence level, or nothing matches, the parser returns
``None`` — it never guesses. Unparsed responses are counted in the
report's unparsed rate.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

_MARKERS = ("label", "answer", "sentiment", "prediction", "class")


def _whole_word_pattern(label: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])" + re.escape(label) + r"(?![\w-])", re.IGNORECASE)


def parse_label(text: str, labels: Sequence[str]) -> Optional[str]:
    """Extract one label from a generated response, or ``None``."""
    if not isinstance(text, str) or not text.strip():
        return None
    pats = {lab: _whole_word_pattern(lab) for lab in labels}

    # 1. explicit marker
    for marker in _MARKERS:
        m = re.search(
            rf"\b{marker}\b\s*[:\-]\s*([^\n.,;]+)", text, re.IGNORECASE
        )
        if m:
            span = m.group(1)
            hits = [lab for lab in labels if pats[lab].search(span)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                return None

    # 2. leading label
    stripped = text.strip()
    hits = [
        lab
        for lab in labels
        if pats[lab].match(stripped)
        and (
            len(stripped) == len(lab)
            or not (stripped[len(lab)].isalnum() or stripped[len(lab)] in "_-")
        )
    ]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None

    # 3. bare label in a short response
    if len(stripped) <= 40:
        hits = [lab for lab in labels if pats[lab].search(stripped)]
        if len(hits) == 1:
            return hits[0]

    return None
