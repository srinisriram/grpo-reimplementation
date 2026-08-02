from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{\s*([^{}]+?)\s*\}")


def extract_final_answer(text: str) -> str | None:
    matches = _BOXED.findall(text)
    return matches[-1].strip() if matches else None


def gsm8k_reward(completion: str, reference_answer: str) -> float:
    """Conservative exact-match scaffold; record any normalization change in decisions.md."""
    predicted = extract_final_answer(completion)
    target = reference_answer.split("####")[-1].strip()
    return float(predicted == target)

