"""Evaluate the base model plus every checkpoint under a run's output_dir, print one table.

Usage: python scripts/compare_checkpoints.py --config configs/full_run.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from grpo.config import load_config
from grpo.evaluate import DEFAULT_NUM_EXAMPLES, evaluate
from grpo.modeling import resolve_device


def _discover_checkpoints(output_dir: str) -> list[Path]:
    def step(path: Path) -> int:
        return int(re.search(r"checkpoint-(\d+)", path.name).group(1))

    return sorted(Path(output_dir).glob("checkpoint-*"), key=step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model vs. every checkpoint in a run")
    parser.add_argument("--config", default="configs/full_run.yaml")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    num_examples = args.num_examples or config.max_eval_examples or DEFAULT_NUM_EXAMPLES
    device = resolve_device()

    checkpoints = _discover_checkpoints(config.output_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* directories found under {config.output_dir}")

    rows: list[tuple[str, dict]] = []

    print("Evaluating base model (no adapter)...")
    rows.append(("base", evaluate(config, None, num_examples, device)))

    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint.name}...")
        rows.append((checkpoint.name, evaluate(config, str(checkpoint), num_examples, device)))

    print()
    print(f"{'checkpoint':<16}{'accuracy':>10}{'correct':>10}{'mean_len':>10}{'correct_len':>13}{'wrong_len':>11}")
    print("-" * 70)
    for name, r in rows:
        correct_len = f"{r['mean_correct_length']:.1f}" if r["mean_correct_length"] is not None else "n/a"
        wrong_len = f"{r['mean_incorrect_length']:.1f}" if r["mean_incorrect_length"] is not None else "n/a"
        print(
            f"{name:<16}{r['accuracy']:>10.3f}{r['num_correct']:>6}/{r['num_examples']:<4}"
            f"{r['mean_completion_length']:>10.1f}{correct_len:>13}{wrong_len:>11}"
        )


if __name__ == "__main__":
    main()
