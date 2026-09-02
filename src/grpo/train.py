from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from .config import TrainingConfig, load_config
from .data import format_prompt, load_gsm8k
from .modeling import load_policy
from .objective import completion_logprobs, group_relative_advantages, grpo_loss
from .rewards import gsm8k_reward
from .rollouts import RolloutBatch, generate_groups


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps,
        )
    if config.lr_scheduler == "cosine":
        min_lr_ratio = config.min_learning_rate / config.learning_rate if config.learning_rate else 0.0

        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler!r} (expected 'linear' or 'cosine')")


def _reduce_mean(accelerator: Accelerator, value: float) -> float:
    # Collective op: every process must call this, even though only the main
    # process ends up printing the result, or processes that skip it will hang
    # waiting for the others to participate in the all-reduce.
    tensor = torch.tensor(value, device=accelerator.device)
    return accelerator.reduce(tensor, reduction="mean").item()


def _forward_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return completion_logprobs(logits, input_ids)


def _sample_prompt_examples(examples: list[dict], cursor: int, batch_size: int) -> list[dict]:
    n = len(examples)
    return [examples[(cursor + i) % n] for i in range(batch_size)]


def _rollout_rewards(rollouts: RolloutBatch, prompt_examples: list[dict], group_size: int) -> torch.Tensor:
    values = []
    for i in range(rollouts.prompt_count):
        reference_answer = prompt_examples[i]["answer"]
        group_texts = rollouts.texts[i * group_size:(i + 1) * group_size]
        values.extend(gsm8k_reward(text, reference_answer) for text in group_texts)
    return torch.tensor(values, dtype=torch.float32)


def _broadcast_advantages(rewards: torch.Tensor, prompt_count: int, group_size: int, num_tokens: int) -> torch.Tensor:
    per_rollout = group_relative_advantages(rewards.view(prompt_count, group_size)).reshape(-1)
    return per_rollout.unsqueeze(-1).expand(-1, num_tokens)


def _save_checkpoint(accelerator: Accelerator, model, output_dir: str, tag: str) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path = Path(output_dir) / f"checkpoint-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path)


def _build_micro_batch(accelerator: Accelerator, model, tokenizer, prompt_examples: list[dict], config: TrainingConfig) -> dict:
    device = accelerator.device
    unwrapped = accelerator.unwrap_model(model)
    prompts = [format_prompt(example["question"]) for example in prompt_examples]
    rollouts = generate_groups(
        unwrapped, tokenizer, prompts, group_size=config.group_size, max_new_tokens=config.max_new_tokens,
    )
    rewards = _rollout_rewards(rollouts, prompt_examples, config.group_size).to(device)
    advantages = _broadcast_advantages(
        rewards, rollouts.prompt_count, config.group_size, rollouts.completion_mask.shape[1]
    )
    with torch.no_grad():
        old_logps = _forward_logps(model, rollouts.input_ids, rollouts.attention_mask)
    with torch.no_grad(), unwrapped.disable_adapter():
        ref_logps = _forward_logps(model, rollouts.input_ids, rollouts.attention_mask)
    return {
        "input_ids": rollouts.input_ids,
        "attention_mask": rollouts.attention_mask,
        "completion_mask": rollouts.completion_mask,
        "advantages": advantages,
        "old_logps": old_logps,
        "ref_logps": ref_logps,
        "mean_reward": rewards.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training loop")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # static_graph=True: gradient checkpointing recomputes parts of the forward
    # pass during backward, which can confuse DDP's default gradient-sync hook
    # ordering. Without this, DDP's bucket-rebuild heuristic can silently lock
    # onto a broken sync configuration a few iterations in -- forward-pass loss
    # values keep looking normal, but the synced gradient reaching the optimizer
    # stops actually updating the weights. static_graph tells DDP the same
    # parameters need gradients every iteration, which is true here (same
    # architecture every step) and is the documented fix for this interaction.
    ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    model, tokenizer = load_policy(config, device=str(accelerator.device))

    train_examples = list(load_gsm8k("train", config=config.dataset_config, limit=config.max_train_examples))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(
        f"Loaded {len(train_examples)} examples and {sum(p.numel() for p in trainable_params):,} trainable "
        f"parameters across {accelerator.num_processes} process(es)."
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    scheduler = _build_scheduler(optimizer, config)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    save_steps = config.save_steps or config.max_steps
    # Each process must see a disjoint slice of the data: process i starts
    # per_device_batch_size*i examples in, and every draw (across all processes
    # combined) advances the shared "virtual" cursor by per_device_batch_size *
    # num_processes, so no two processes ever train on the same prompts.
    cursor = config.per_device_batch_size * accelerator.process_index
    stride = config.per_device_batch_size * accelerator.num_processes

    for step in range(1, config.max_steps + 1):
        model.eval()
        micro_batches = []
        for _ in range(config.gradient_accumulation_steps):
            prompt_examples = _sample_prompt_examples(train_examples, cursor, config.per_device_batch_size)
            cursor += stride
            micro_batches.append(_build_micro_batch(accelerator, model, tokenizer, prompt_examples, config))

        model.train()
        last_metrics: dict[str, torch.Tensor] = {}
        for _ in range(config.ppo_epochs):
            optimizer.zero_grad()
            for mb in micro_batches:
                current_logps = _forward_logps(model, mb["input_ids"], mb["attention_mask"])
                loss, metrics = grpo_loss(
                    current_logps, mb["old_logps"], mb["ref_logps"], mb["advantages"],
                    mb["completion_mask"], clip_range=config.clip_range, kl_coef=config.kl_coef,
                )
                accelerator.backward(loss / config.gradient_accumulation_steps)
                last_metrics = metrics
            optimizer.step()
            scheduler.step()

        local_mean_reward = sum(mb["mean_reward"] for mb in micro_batches) / len(micro_batches)
        # Each process only ever sees its own local slice of the step's data;
        # reduce across all processes so the printed numbers reflect the full
        # combined batch, not just whichever process happens to be main.
        mean_reward = _reduce_mean(accelerator, local_mean_reward)
        policy_loss = _reduce_mean(accelerator, last_metrics["policy_loss"].item())
        kl = _reduce_mean(accelerator, last_metrics["kl"].item())
        clip_fraction = _reduce_mean(accelerator, last_metrics["clip_fraction"].item())
        accelerator.print(
            f"step {step}/{config.max_steps} reward={mean_reward:.3f} "
            f"policy_loss={policy_loss:.4f} "
            f"kl={kl:.4f} "
            f"clip_fraction={clip_fraction:.3f}"
        )

        if step % save_steps == 0 or step == config.max_steps:
            _save_checkpoint(accelerator, model, config.output_dir, str(step))

    accelerator.print(f"Training complete. Checkpoints in {config.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
