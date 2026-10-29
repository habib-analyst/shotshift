# Changelog

## 0.1.0 — 2026-10-29

Initial release.

- `probe()` — run a score-mode or generate-mode model callable over a set
  of example orders × queries and get a `ShotReport`: stability score,
  flip rate, fragile-query rate, per-query cards, canonical / mean / min /
  max accuracy and accuracy range over orders, majority-vote accuracy,
  predicted-label distribution, unparsed rate (generate mode).
- `probe_zero_shot()` — the same machinery with no examples, for a
  like-for-like zero-shot comparison.
- `leave_one_out()` — re-probe with each example removed in turn and
  report each example's delta to mean accuracy and stability.
- Order sets are exhaustive when the permutation budget allows and a
  seeded sample otherwise; the exact set, the seed, and the prompt
  templates are recorded in every report's config.
- Measured demo: Qwen2.5-0.5B-Instruct on SST-2 sentiment classification
  with an 8-example prompt over 16 orders (see README).
