"""Measured demo — a real model on real data.

Qwen2.5-0.5B-Instruct (Apache-2.0) classifies SST-2 sentiment with an
8-example few-shot prompt. ShotShift probes 16 orders of the examples
over the first 48 validation queries (plus a zero-shot pass and a
leave-one-out pass at 4 orders), scoring by the next-token logprob of
" positive" / " negative" after the prompt.

Data subsets are deterministic file-order slices of the published
parquets: demos are the first 4 negative and first 4 positive *training*
rows, merged in file order; queries are the first 48 *validation* rows.
Nothing is cherry-picked; a demo sentence whose wording looks at odds
with its label carries the dataset's own label, as published.

Execution is two-phase because this VM is slow and kills long jobs:
  --phase score     order-major scoring into a resumable JSONL cache
                    (prefix KV encoded once per order); safe to re-run,
                    skips items already cached; --limit N caps new items
  --phase assemble  build the reports through shotshift.probe /
                    leave_one_out backed by the cache, verify every
                    prompt the library renders was actually scored, and
                    write demo_summary.json + demo_report.json

The scoring path is parity-checked: the first main-config item is also
scored with a plain full forward (no KV reuse) and the agreement is
recorded in the summary. The check exists because this transformers
version silently corrupts KV-path logits when the continuation is
passed with a continuation-length attention mask, and because a reused
KV cache must be cropped back to the prefix length after every item —
both failure modes were caught by the parity gate during this build
(see REPORT.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotshift import Example, Query, leave_one_out, probe, probe_zero_shot  # noqa: E402
from shotshift.prompts import make_orders, render_prompt  # noqa: E402

SEED = 20261029
MAIN_ORDERS = 16
LOO_ORDERS = 4
N_QUERIES = 48
LABELS = ["positive", "negative"]
INSTRUCTION = "Classify the sentiment of the review excerpt."
EXAMPLE_TEMPLATE = "Review: {text}\nSentiment: {label}"
QUERY_TEMPLATE = "Review: {text}\nSentiment:"
SEPARATOR = "\n\n"
RENDER_KW = dict(
    instruction=INSTRUCTION,
    example_template=EXAMPLE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
    separator=SEPARATOR,
)
OUT = ROOT / "examples" / "output"
CACHE = OUT / "score_cache.jsonl"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_subsets(data_dir: Path):
    import pyarrow.parquet as pq

    train = pq.read_table(data_dir / "sst2_train.parquet").to_pylist()
    val = pq.read_table(data_dir / "sst2_validation.parquet").to_pylist()
    name = {0: "negative", 1: "positive"}
    chosen: list[tuple[int, dict]] = []
    for want in (0, 1):
        picked = [(i, r) for i, r in enumerate(train) if r["label"] == want][:4]
        chosen.extend(picked)
    chosen.sort(key=lambda t: t[0])  # file order across both classes
    examples = [Example(r["sentence"].strip(), name[r["label"]]) for _, r in chosen]
    demo_record = [
        {"train_index": i, "label": name[r["label"]], "text": r["sentence"].strip()}
        for i, r in chosen
    ]
    queries = [
        Query(r["sentence"].strip(), answer=name[r["label"]])
        for r in val[:N_QUERIES]
    ]
    return examples, queries, demo_record


class Engine:
    """Prefix-cached label scorer over the local Qwen weights."""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(2)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32
        )
        self.model.eval()
        self.label_ids = {}
        for lab in LABELS:
            ids = self.tok.encode(" " + lab)
            assert len(ids) == 1, f"label continuation {lab!r} is not one token: {ids}"
            self.label_ids[lab] = ids[0]
        self._prefix_cache: dict[str, tuple[object, int]] = {}

    def _prefix_kv(self, prefix_text: str):
        if prefix_text not in self._prefix_cache:
            ids = self.tok(prefix_text, return_tensors="pt")
            with self.torch.no_grad():
                out = self.model(**ids, use_cache=True)
            n_tokens = int(ids["input_ids"].shape[1])
            self._prefix_cache = {prefix_text: (out.past_key_values, n_tokens)}
        return self._prefix_cache[prefix_text]

    def score(self, prefix_text: str, query_text: str) -> dict[str, float]:
        """Logprobs of the label continuations after prefix + query block.

        Two hard-won details (see the module docstring): the continuation
        is passed as input_ids ONLY — a continuation-length attention
        mask corrupts the logits on this stack — and the shared prefix
        cache is cropped back to the prefix length after every call,
        because the model appends the continuation to it in place.
        """
        torch = self.torch
        kv, n_prefix = self._prefix_kv(prefix_text)
        q_ids = self.tok(query_text, return_tensors="pt")
        with torch.no_grad():
            out = self.model(
                input_ids=q_ids["input_ids"], past_key_values=kv, use_cache=True
            )
        logits = out.logits[0, -1]
        if hasattr(kv, "crop"):
            kv.crop(n_prefix)
        logprobs = torch.log_softmax(logits, dim=-1)
        return {lab: float(logprobs[tid]) for lab, tid in self.label_ids.items()}

    def score_plain(self, prompt: str) -> dict[str, float]:
        """Full forward, no KV reuse — the parity reference."""
        torch = self.torch
        ids = self.tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**ids)
        logits = out.logits[0, -1]
        logprobs = torch.log_softmax(logits, dim=-1)
        return {lab: float(logprobs[tid]) for lab, tid in self.label_ids.items()}


def split_prompt(prompt: str) -> tuple[str, str]:
    prefix, _, query_block = prompt.rpartition(SEPARATOR)
    return prefix, query_block


def work_items(examples, queries):
    """All (key, prompt) items to score, order-major within each config."""
    items: list[tuple[str, str]] = []

    def add(tag: str, exs, orders) -> None:
        for oi, order in enumerate(orders):
            for qi, q in enumerate(queries):
                items.append((f"{tag}|{oi}|{qi}", render_prompt(exs, order, q, **RENDER_KW)))

    add("main", examples, make_orders(len(examples), max_orders=MAIN_ORDERS, seed=SEED))
    for qi, q in enumerate(queries):
        items.append((f"zero|0|{qi}", render_prompt([], (), q, **RENDER_KW)))
    loo_orders = make_orders(len(examples) - 1, max_orders=LOO_ORDERS, seed=SEED)
    for removed in range(len(examples)):
        reduced = examples[:removed] + examples[removed + 1:]
        add(f"loo{removed}", reduced, loo_orders)
    return items


def phase_score(model_dir: Path, data_dir: Path, limit: int | None) -> None:
    examples, queries, _ = load_subsets(data_dir)
    items = work_items(examples, queries)
    done: dict[str, dict] = {}
    if CACHE.exists():
        for line in open(CACHE):
            rec = json.loads(line)
            done[rec["key"]] = rec["scores"]
    todo = [(k, p) for k, p in items if k not in done]
    print(f"cache holds {len(done)}/{len(items)} items; {len(todo)} to score", flush=True)
    if not todo:
        print("SCORE_COMPLETE")
        return
    engine = Engine(model_dir)
    parity_path = OUT / "parity_check.json"
    if "main|0|0" in done and parity_path.exists():
        parity = json.loads(parity_path.read_text())
        if "repeat_check" not in parity:
            prompt00 = dict(items)["main|0|0"]
            prefix, query_block = split_prompt(prompt00)
            tail = (SEPARATOR + query_block) if prefix else query_block
            fresh = engine.score(prefix, tail)
            cached = done["main|0|0"]
            parity["repeat_check"] = {
                "argmax_agree": max(fresh, key=fresh.get) == max(cached, key=cached.get),
                "max_abs_logprob_delta": max(
                    abs(fresh[lab] - cached[lab]) for lab in LABELS
                ),
            }
            parity_path.write_text(json.dumps(parity, indent=2))
            print("repeat check:", parity["repeat_check"], flush=True)
    t0 = time.time()
    n = 0
    with open(CACHE, "a") as f:
        for key, prompt in todo:
            if limit is not None and n >= limit:
                break
            prefix, query_block = split_prompt(prompt)
            tail = (SEPARATOR + query_block) if prefix else query_block
            scores = engine.score(prefix, tail)
            f.write(json.dumps({"key": key, "scores": scores}) + "\n")
            f.flush()
            n += 1
            if key == "main|0|0":
                plain = engine.score_plain(prompt)
                parity = {
                    "argmax_agree": max(scores, key=scores.get) == max(plain, key=plain.get),
                    "max_abs_logprob_delta": max(
                        abs(scores[lab] - plain[lab]) for lab in LABELS
                    ),
                }
                (OUT / "parity_check.json").write_text(json.dumps(parity, indent=2))
                print("parity:", parity, flush=True)
    print(f"scored {n} items in {time.time() - t0:.1f}s; cache now {len(done) + n}/{len(items)}")
    if len(done) + n >= len(items):
        print("SCORE_COMPLETE")


def phase_assemble(model_dir: Path, data_dir: Path) -> None:
    examples, queries, demo_record = load_subsets(data_dir)
    items = dict(work_items(examples, queries))
    cache: dict[str, dict] = {}
    for line in open(CACHE):
        rec = json.loads(line)
        cache[rec["key"]] = rec["scores"]
    if len(cache) < len(items):
        raise SystemExit(f"cache incomplete: {len(cache)}/{len(items)} — run --phase score")
    by_prompt: dict[str, dict] = {}
    for key, prompt in items.items():
        if prompt in by_prompt and by_prompt[prompt] != cache[key]:
            raise AssertionError(f"conflicting scores for one prompt (key {key})")
        by_prompt[prompt] = cache[key]

    # Independent audit: recompute a spread of cached items with plain
    # full forwards (no KV path at all) and compare against the cache.
    engine = Engine(model_dir)
    cache_audit = []
    for key in ["main|0|0", "main|0|47", "main|7|13", "loo3|2|10", "zero|0|5"]:
        plain = engine.score_plain(items[key])
        cached = cache[key]
        cache_audit.append(
            {
                "key": key,
                "argmax_agree": max(plain, key=plain.get) == max(cached, key=cached.get),
                "max_abs_logprob_delta": max(
                    abs(plain[lab] - cached[lab]) for lab in LABELS
                ),
            }
        )
    print("cache audit:", cache_audit, flush=True)

    def cached_fn(prompt: str) -> dict[str, float]:
        if prompt not in by_prompt:
            raise AssertionError("library rendered a prompt that was never scored")
        return by_prompt[prompt]

    main = probe(cached_fn, examples, queries,
                 max_orders=MAIN_ORDERS, seed=SEED, **RENDER_KW)
    zero = probe_zero_shot(
        cached_fn, LABELS, queries,
        instruction=INSTRUCTION, query_template=QUERY_TEMPLATE, separator=SEPARATOR,
    )
    loo = leave_one_out(cached_fn, examples, queries,
                        max_orders=LOO_ORDERS, seed=SEED, **RENDER_KW)
    parity = json.loads((OUT / "parity_check.json").read_text())
    summary = {
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "license": "Apache-2.0",
            "local_safetensors_sha256": sha256_of(model_dir / "model.safetensors"),
            "scoring": "next-token logprob of ' positive' / ' negative' (one token each, asserted)",
        },
        "data": {
            "dataset": "SST-2 (GLUE benchmark distribution: stanfordnlp/sst2 on Hugging Face)",
            "train_parquet_sha256": sha256_of(data_dir / "sst2_train.parquet"),
            "validation_parquet_sha256": sha256_of(data_dir / "sst2_validation.parquet"),
            "subset_rule": "demos = first 4 negative + first 4 positive train rows, merged in file order; queries = first 48 validation rows in file order",
            "demo_examples": demo_record,
            "query_label_counts": {
                lab: sum(1 for q in queries if q.answer == lab) for lab in LABELS
            },
        },
        "config": {
            "seed": SEED,
            "main_orders": MAIN_ORDERS,
            "loo_orders": LOO_ORDERS,
            "n_queries": len(queries),
            "templates": {
                "instruction": INSTRUCTION,
                "example_template": EXAMPLE_TEMPLATE,
                "query_template": QUERY_TEMPLATE,
                "separator": SEPARATOR,
            },
        },
        "parity_check": parity,
        "cache_audit": cache_audit,
        "main_report": main.to_dict()["metrics"],
        "accuracy_by_order": main.accuracy_by_order(),
        "zero_shot": zero.to_dict()["metrics"],
        "leave_one_out": [
            {
                "removed_index": e["removed_index"],
                "removed_label": e["removed_example"]["label"],
                "delta_mean_accuracy": e["delta_mean_accuracy"],
                "delta_stability": e["delta_stability"],
            }
            for e in loo["leave_one_out"]
        ],
        "most_fragile_queries": [
            {
                "query_index": c.query_index,
                "answer": c.answer,
                "canonical": c.canonical,
                "modal": c.modal,
                "modal_share": c.modal_share,
                "n_flips": c.n_flips,
            }
            for c in sorted(main.cards, key=lambda c: c.modal_share)[:10]
        ],
    }
    (OUT / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "demo_report.json").write_text(json.dumps(main.to_dict(), indent=2))
    print(main.summary())
    print("zero-shot canonical accuracy:", zero.canonical_accuracy)
    print("ASSEMBLE_COMPLETE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["score", "assemble"], required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "work_assets" / "model")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "work_assets" / "data")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "score":
        phase_score(args.model, args.data_dir, args.limit)
    else:
        phase_assemble(args.model, args.data_dir)


if __name__ == "__main__":
    main()
