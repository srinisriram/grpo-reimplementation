from __future__ import annotations

from datasets import load_dataset


SYSTEM_PROMPT = "Solve the problem. Put only the final numerical answer in \\boxed{}."


def format_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def load_gsm8k(split: str, *, config: str = "main", limit: int | None = None):
    dataset = load_dataset("openai/gsm8k", config, split=split)
    return dataset.select(range(min(limit, len(dataset)))) if limit else dataset

