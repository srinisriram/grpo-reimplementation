"""Analyze a per-example completions dump from `evaluate.py --dump_completions`.

Usage: python scripts/analyze_completions.py outputs/completions_checkpoint-80.jsonl

Prints the report live and also saves it to a file (by default, alongside the
dump as <dump_path>.analysis.txt), so a long-scrollback screen/tmux session
running several of these back to back doesn't lose earlier output.
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
    parser.add_argument(
        "--output", default=None,
        help="Where to save the report (default: <dump_path>.analysis.txt). Pass '' to skip saving.",
    )
    args = parser.parse_args()

    lines = [line for line in Path(args.dump_path).read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    total = len(records)
    correct = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]

    report: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        report.append(text)

    emit(f"{args.dump_path}: {total} examples, {len(correct)} correct, {len(wrong)} wrong")
    emit()

    emit("=== (a) Truncation check ===")
    wrong_hit_cap = [r for r in wrong if r["hit_token_cap"]]
    wrong_no_answer = [r for r in wrong if r["predicted_answer"] is None]
    wrong_no_answer_and_capped = [r for r in wrong_no_answer if r["hit_token_cap"]]
    if wrong:
        emit(f"Wrong completions that hit the token cap: {len(wrong_hit_cap)}/{len(wrong)} "
             f"({100 * len(wrong_hit_cap) / len(wrong):.1f}%)")
        emit(f"Wrong completions with no \\boxed{{}} answer parsed at all: {len(wrong_no_answer)}/{len(wrong)} "
             f"({100 * len(wrong_no_answer) / len(wrong):.1f}%)")
        if wrong_no_answer:
            emit(f"  ...of those, also hit the token cap (strongest truncation signal): "
                 f"{len(wrong_no_answer_and_capped)}/{len(wrong_no_answer)} "
                 f"({100 * len(wrong_no_answer_and_capped) / len(wrong_no_answer):.1f}%)")
    else:
        emit("No wrong completions.")
    emit()

    emit("=== (b) Length vs correctness ===")
    correct_lengths = [r["length"] for r in correct]
    wrong_lengths = [r["length"] for r in wrong]
    if correct_lengths:
        emit(f"Correct: mean={statistics.mean(correct_lengths):.1f}, "
             f"median={statistics.median(correct_lengths):.1f}, "
             f"min={min(correct_lengths)}, max={max(correct_lengths)}")
    if wrong_lengths:
        emit(f"Wrong:   mean={statistics.mean(wrong_lengths):.1f}, "
             f"median={statistics.median(wrong_lengths):.1f}, "
             f"min={min(wrong_lengths)}, max={max(wrong_lengths)}")
    emit()

    n = min(args.num_samples, len(wrong))
    emit(f"=== (c) {n} sample wrong completions ===")
    for i, r in enumerate(wrong[:n], start=1):
        emit(f"\n--- wrong example {i} "
             f"(length={r['length']}, hit_cap={r['hit_token_cap']}, "
             f"predicted={r['predicted_answer']!r}, reference={r['reference_answer']!r}) ---")
        emit(f"Q: {r['question']}")
        emit(f"Completion: {r['completion']}")

    output_path = args.output if args.output is not None else f"{args.dump_path}.analysis.txt"
    if output_path:
        Path(output_path).write_text("\n".join(report) + "\n")
        print(f"\n(report saved to {output_path})")


if __name__ == "__main__":
    main()
