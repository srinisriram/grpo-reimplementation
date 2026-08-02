from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    output_dir: str
    per_device_batch_size: int
    group_size: int
    max_new_tokens: int
    learning_rate: float
    max_steps: int
    gradient_accumulation_steps: int
    kl_coef: float
    clip_range: float
    seed: int
    lora_rank: int
    lora_alpha: int
    target_modules: list[str]
    dataset: str
    dataset_config: str
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    ppo_epochs: int = 1
    save_steps: int | None = None
    warmup_steps: int = 0


def load_config(path: str | Path) -> TrainingConfig:
    """Load the small YAML configuration format used by the entry points."""
    with Path(path).open() as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    model, data, training = raw["model"], raw["data"], raw["training"]
    return TrainingConfig(
        model_name=model["name"], output_dir=training["output_dir"],
        per_device_batch_size=training["per_device_batch_size"], group_size=training["group_size"],
        max_new_tokens=training["max_new_tokens"], learning_rate=training["learning_rate"],
        max_steps=training["max_steps"], gradient_accumulation_steps=training["gradient_accumulation_steps"],
        kl_coef=training["kl_coef"], clip_range=training["clip_range"], seed=training["seed"],
        lora_rank=model["lora_rank"], lora_alpha=model["lora_alpha"],
        target_modules=model["target_modules"], dataset=data["dataset"],
        dataset_config=data["config"], max_train_examples=data.get("max_train_examples"),
        max_eval_examples=data.get("max_eval_examples"),
        ppo_epochs=training.get("ppo_epochs", 1), save_steps=training.get("save_steps"),
        warmup_steps=training.get("warmup_steps", 0),
    )

