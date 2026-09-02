"""Analyze a per-example completions dump from `evaluate.py --dump_completions`.

Usage: python scripts/analyze_completions.py outputs/completions_checkpoint-80.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completions dump for truncation/length effects")
    parser.add_argument("dump_path")
    parser.add_argument("--num_samples", type=int, default=5, help="How many wrong completions to print in full")
    args = parser.parse_args()

    lines = [line for line in Path(args.dump_path).read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    total = len(records)
    correct = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]

    print(f"{args.dump_path}: {total} examples, {len(correct)} correct, {len(wrong)} wrong")
    print()

    print("=== (a) Truncation check ===")
    wrong_hit_cap = [r for r in wrong if r["hit_token_cap"]]
    wrong_no_answer = [r for r in wrong if r["predicted_answer"] is None]
    wrong_no_answer_and_capped = [r for r in wrong_no_answer if r["hit_token_cap"]]
    print(f"Wrong completions that hit the token cap: {len(wrong_hit_cap)}/{len(wrong)} "
          f"({100 * len(wrong_hit_cap) / len(wrong):.1f}%)" if wrong else "No wrong completions.")
    print(f"Wrong completions with no \\boxed{{}} answer parsed at all: {len(wrong_no_answer)}/{len(wrong)} "
          f"({100 * len(wrong_no_answer) / len(wrong):.1f}%)" if wrong else "")
    if wrong_no_answer:
        print(f"  ...of those, also hit the token cap (strongest truncation signal): "
              f"{len(wrong_no_answer_and_capped)}/{len(wrong_no_answer)} "
              f"({100 * len(wrong_no_answer_and_capped) / len(wrong_no_answer):.1f}%)")
    print()

    print("=== (b) Length vs correctness ===")
    correct_lengths = [r["length"] for r in correct]
    wrong_lengths = [r["length"] for r in wrong]
    if correct_lengths:
        print(f"Correct: mean={statistics.mean(correct_lengths):.1f}, "
              f"median={statistics.median(correct_lengths):.1f}, "
              f"min={min(correct_lengths)}, max={max(correct_lengths)}")
    if wrong_lengths:
        print(f"Wrong:   mean={statistics.mean(wrong_lengths):.1f}, "
              f"median={statistics.median(wrong_lengths):.1f}, "
              f"min={min(wrong_lengths)}, max={max(wrong_lengths)}")
    print()

    n = min(args.num_samples, len(wrong))
    print(f"=== (c) {n} sample wrong completions ===")
    for i, r in enumerate(wrong[:n], start=1):
        print(f"\n--- wrong example {i} "
              f"(length={r['length']}, hit_cap={r['hit_token_cap']}, "
              f"predicted={r['predicted_answer']!r}, reference={r['reference_answer']!r}) ---")
        print(f"Q: {r['question']}")
        print(f"Completion: {r['completion']}")


if __name__ == "__main__":
    main()
