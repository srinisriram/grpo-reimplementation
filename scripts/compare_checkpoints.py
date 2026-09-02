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


def _format_row(name: str, r: dict) -> str:
    correct_len = f"{r['mean_correct_length']:.1f}" if r["mean_correct_length"] is not None else "n/a"
    wrong_len = f"{r['mean_incorrect_length']:.1f}" if r["mean_incorrect_length"] is not None else "n/a"
    return (
        f"{name:<16}{r['accuracy']:>10.3f}{r['num_correct']:>6}/{r['num_examples']:<4}"
        f"{r['mean_completion_length']:>10.1f}{correct_len:>13}{wrong_len:>11}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model vs. every checkpoint in a run")
    parser.add_argument("--config", default="configs/full_run.yaml")
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument(
        "--dump_dir", default=None,
        help="If set, write per-example completions for every evaluated model to "
             "<dump_dir>/completions_<name>.jsonl (base and each checkpoint).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    num_examples = args.num_examples or config.max_eval_examples or DEFAULT_NUM_EXAMPLES
    device = resolve_device()
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _discover_checkpoints(config.output_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* directories found under {config.output_dir}")

    header = f"{'checkpoint':<16}{'accuracy':>10}{'correct':>10}{'mean_len':>10}{'correct_len':>13}{'wrong_len':>11}"
    rows: list[tuple[str, dict]] = []

    def run(name: str, checkpoint: str | None) -> None:
        print(f"\n=== Evaluating {name} ===", flush=True)
        dump_path = str(dump_dir / f"completions_{name}.jsonl") if dump_dir is not None else None
        result = evaluate(config, checkpoint, num_examples, device, dump_path=dump_path)
        rows.append((name, result))
        print(f"  done: {_format_row(name, result)}", flush=True)

    run("base", None)
    for checkpoint in checkpoints:
        run(checkpoint.name, str(checkpoint))

    print("\n" + header)
    print("-" * len(header))
    for name, r in rows:
        print(_format_row(name, r))


if __name__ == "__main__":
    main()
