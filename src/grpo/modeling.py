from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TrainingConfig


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, device: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)


def load_policy(config: TrainingConfig):
    tokenizer = load_tokenizer(config.model_name)
    model = load_base_model(config.model_name, resolve_device())
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_rank, lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
    )
    return get_peft_model(model, lora), tokenizer

