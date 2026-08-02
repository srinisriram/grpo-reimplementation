from __future__ import annotations

import argparse

from .config import load_config
from .data import load_gsm8k


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_gsm8k("test", config=config.dataset_config, limit=config.max_eval_examples)
    print(f"Evaluation scaffold: {len(dataset)} examples; checkpoint={args.checkpoint}")
    print("Add deterministic generation, answer parsing metrics, and JSONL predictions here.")


if __name__ == "__main__":
    main()

