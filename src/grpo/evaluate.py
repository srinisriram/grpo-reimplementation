from __future__ import annotations

import argparse
import time

import torch
from peft import PeftModel

from .config import TrainingConfig, load_config
from .data import format_prompt, load_gsm8k
from .modeling import load_base_model, load_tokenizer, resolve_device
from .rewards import gsm8k_reward

# Fixed independent of any training config's own seed, so every evaluation call
# (baseline or any checkpoint) scores the exact same 100 test problems.
EVAL_SEED = 1234
DEFAULT_NUM_EXAMPLES = 100
GENERATION_BATCH_SIZE = 8


def _load_eval_model(model_name: str, checkpoint: str | None, device: str):
    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name, device)
    model = base_model if checkpoint is None else PeftModel.from_pretrained(base_model, checkpoint).to(device)
    model.eval()
    return model, tokenizer


def _generate_deterministic(
    model, tokenizer, prompts: list[str], max_new_tokens: int, device: str
) -> tuple[list[str], list[int]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_width = encoded.input_ids.shape[1]
    with torch.no_grad():
        generated = model.generate(
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
            do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = generated[:, prompt_width:]
    lengths = (completion_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return texts, lengths


def evaluate(config: TrainingConfig, checkpoint: str | None, num_examples: int, device: str, verbose: bool = True) -> dict:
    if verbose:
        print(f"  loading model{f' + adapter {checkpoint}' if checkpoint else ' (base, no adapter)'}...", flush=True)
    load_start = time.time()
    model, tokenizer = _load_eval_model(config.model_name, checkpoint, device)
    examples = list(
        load_gsm8k("test", config=config.dataset_config).shuffle(seed=EVAL_SEED).select(range(num_examples))
    )
    if verbose:
        print(f"  model loaded in {time.time() - load_start:.1f}s, generating on {len(examples)} examples...", flush=True)

    rewards: list[float] = []
    lengths: list[int] = []
    num_batches = (len(examples) + GENERATION_BATCH_SIZE - 1) // GENERATION_BATCH_SIZE
    for batch_idx, start in enumerate(range(0, len(examples), GENERATION_BATCH_SIZE), start=1):
        batch_start = time.time()
        batch = examples[start:start + GENERATION_BATCH_SIZE]
        prompts = [format_prompt(example["question"]) for example in batch]
        completions, batch_lengths = _generate_deterministic(model, tokenizer, prompts, config.max_new_tokens, device)
        for example, completion in zip(batch, completions):
            rewards.append(gsm8k_reward(completion, example["answer"]))
        lengths.extend(batch_lengths)
        if verbose:
            print(
                f"    batch {batch_idx}/{num_batches} ({len(rewards)}/{len(examples)} examples, "
                f"{time.time() - batch_start:.1f}s) — running accuracy: {int(sum(rewards))}/{len(rewards)}",
                flush=True,
            )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    correct_lengths = [length for length, reward in zip(lengths, rewards) if reward == 1.0]
    incorrect_lengths = [length for length, reward in zip(lengths, rewards) if reward == 0.0]
    return {
        "num_examples": num_examples,
        "num_correct": int(sum(rewards)),
        "accuracy": sum(rewards) / len(rewards),
        "mean_completion_length": sum(lengths) / len(lengths),
        "mean_correct_length": sum(correct_lengths) / len(correct_lengths) if correct_lengths else None,
        "mean_incorrect_length": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic GSM8K evaluation")
    parser.add_argument("--config", default="configs/full_run.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to a saved LoRA adapter; omit to evaluate the base model")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    num_examples = args.num_examples or config.max_eval_examples or DEFAULT_NUM_EXAMPLES
    device = resolve_device()

    results = evaluate(config, args.checkpoint, num_examples, device)

    label = args.checkpoint or "base model (no adapter)"
    print(f"Evaluated {label} on {results['num_examples']} GSM8K test examples (seed={EVAL_SEED})")
    print(f"  accuracy: {results['accuracy']:.3f} ({results['num_correct']}/{results['num_examples']})")
    print(f"  mean completion length: {results['mean_completion_length']:.1f} tokens")
    if results["mean_correct_length"] is not None:
        print(f"  mean length, correct: {results['mean_correct_length']:.1f} tokens")
    if results["mean_incorrect_length"] is not None:
        print(f"  mean length, incorrect: {results['mean_incorrect_length']:.1f} tokens")


if __name__ == "__main__":
    main()
