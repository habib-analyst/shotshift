# Contributing

Issues and pull requests are welcome on GitHub.

Ground rules, in the spirit of the project:

- **Measurements over opinions.** A change to a metric needs a test that
  pins the new arithmetic on a hand-computable example, the way
  `tests/test_core.py` pins the position-picker and stability cases.
- **No silent guessing.** The parser returns `None` when it cannot read an
  answer; keep it that way. An unparsed rate is a measurement, not an
  embarrassment to be papered over.
- **The library stays dependency-light.** NumPy is the only runtime
  dependency. Model frameworks belong in examples and in users' adapters,
  never in `src/choiceshift`.
- **Determinism.** Anything seeded must be reproducible from the seed
  recorded in the report config.

Run the tests with:

```
python -m pip install -e ".[dev]"
python -m pytest
```
