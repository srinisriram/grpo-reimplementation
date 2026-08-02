from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from .config import TrainingConfig, load_config
from .data import format_prompt, load_gsm8k
from .modeling import load_policy
from .objective import completion_logprobs, group_relative_advantages, grpo_loss
from .rewards import gsm8k_reward
from .rollouts import RolloutBatch, generate_groups


def _forward_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the model's logits from the run, and feed it to the completion_logprobs
    function to extract the model's log-probability of outputting the particular token
    given the tokens that preceded it.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return completion_logprobs(logits, input_ids)


def _sample_prompt_examples(examples: list[dict], cursor: int, batch_size: int) -> tuple[list[dict], int]:
    """
    Grab a batch of prompts for the current batch step, wrapping around by
    cycling back to the start when all the prompts have been used.
    """
    n = len(examples)
    batch = [examples[(cursor + i) % n] for i in range(batch_size)]
    return batch, cursor + batch_size


def _rollout_rewards(rollouts: RolloutBatch, prompt_examples: list[dict], group_size: int) -> torch.Tensor:
    """
    For every prompt group in the microbatch, evaluate their answers according 
    to the reference answer.
    """
    values = []
    for i in range(rollouts.prompt_count):
        reference_answer = prompt_examples[i]["answer"]
        group_texts = rollouts.texts[i * group_size:(i + 1) * group_size]
        values.extend(gsm8k_reward(text, reference_answer) for text in group_texts)
    return torch.tensor(values, dtype=torch.float32)


def _broadcast_advantages(rewards: torch.Tensor, prompt_count: int, group_size: int, num_tokens: int) -> torch.Tensor:
    """
    Apply the group relative advantages for each of the reward scores per 
    sequence in the group, then broadcast each reward advantage score 
    to each token in the respective sequence. 
    """
    per_rollout = group_relative_advantages(rewards.view(prompt_count, group_size)).reshape(-1)
    return per_rollout.unsqueeze(-1).expand(-1, num_tokens)


def _save_checkpoint(model, output_dir: str, tag: str) -> None:
    """
    Save LoRA checkpoints during training.
    """
    path = Path(output_dir) / f"checkpoint-{tag}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)


def _build_micro_batch(model, tokenizer, prompt_examples: list[dict], config: TrainingConfig, device) -> dict:
    """
    Returns a dictionary for all the necessary training values post-generation 
    and reward scoring. Computes and broadcasts advantages, extracts the old
    and ref log-probs, and returns all the necessary training info for one
    epoch on this batch.
    """
    prompts = [format_prompt(example["question"]) for example in prompt_examples]
    rollouts = generate_groups(
        model, tokenizer, prompts, group_size=config.group_size, max_new_tokens=config.max_new_tokens,
    )
    rewards = _rollout_rewards(rollouts, prompt_examples, config.group_size).to(device)
    advantages = _broadcast_advantages(
        rewards, rollouts.prompt_count, config.group_size, rollouts.completion_mask.shape[1]
    )
    # no grad ensures computation graph isn't stored for these log-probs
    with torch.no_grad():
        old_logps = _forward_logps(model, rollouts.input_ids, rollouts.attention_mask)
    with torch.no_grad(), model.disable_adapter():
        ref_logps = _forward_logps(model, rollouts.input_ids, rollouts.attention_mask)
    return {
        "input_ids": rollouts.input_ids, # what ids were actually selected per sequence
        "attention_mask": rollouts.attention_mask,
        "completion_mask": rollouts.completion_mask,  # mask for padding/prompt tokens vs. actual objective tokens
        "advantages": advantages, # advantages broadcasted on a token-basis
        "old_logps": old_logps, # no grad, read the model's stored log-probs pre-update in this round
        "ref_logps": ref_logps, # no grad, read the reference model log-probs stored
        "mean_reward": rewards.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training loop")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    model, tokenizer = load_policy(config)
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    train_examples = list(load_gsm8k("train", config=config.dataset_config, limit=config.max_train_examples))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Loaded {len(train_examples)} examples and {sum(p.numel() for p in trainable_params):,} trainable parameters.")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps,
    )
    save_steps = config.save_steps or config.max_steps
    cursor = 0

    # iterate over a training step
    for step in range(1, config.max_steps + 1):
        # set model to eval mode (just generating batch data)
        model.eval()
        # build the micro batches to learn from
        micro_batches = []
        for _ in range(config.gradient_accumulation_steps):
            prompt_examples, cursor = _sample_prompt_examples(train_examples, cursor, config.per_device_batch_size)
            micro_batches.append(_build_micro_batch(model, tokenizer, prompt_examples, config, device))
        
        # set model to train mode
        model.train()
        last_metrics: dict[str, torch.Tensor] = {}
        for _ in range(config.ppo_epochs):
            # clear old gradients
            optimizer.zero_grad()
            for mb in micro_batches:
                # compute the current log probs with gradient to learn from
                current_logps = _forward_logps(model, mb["input_ids"], mb["attention_mask"])
                # compute the loss and metrics for the particular micro-batch run
                loss, metrics = grpo_loss(
                    current_logps, mb["old_logps"], mb["ref_logps"], mb["advantages"],
                    mb["completion_mask"], clip_range=config.clip_range, kl_coef=config.kl_coef,
                )
                # walk backward and compute the gradient for every LoRA param
                (loss / config.gradient_accumulation_steps).backward()
                last_metrics = metrics
            # optimizer applies the gradient nudges computed via the loss function
            optimizer.step()
            scheduler.step()

        # compute mean reward and output diagnostics
        mean_reward = sum(mb["mean_reward"] for mb in micro_batches) / len(micro_batches)
        print(
            f"step {step}/{config.max_steps} reward={mean_reward:.3f} "
            f"policy_loss={last_metrics['policy_loss'].item():.4f} "
            f"kl={last_metrics['kl'].item():.4f} "
            f"clip_fraction={last_metrics['clip_fraction'].item():.3f}"
        )

        if step % save_steps == 0 or step == config.max_steps:
            # save periodic checkpoints
            _save_checkpoint(model, config.output_dir, str(step))

    print(f"Training complete. Checkpoints in {config.output_dir}")


if __name__ == "__main__":
    """
    walked example flow:
    input_ids/attention_mask : (2, 122)      = [B, L]
    completion_mask           : (2, 121)      = [B, T]
    rewards                   : (2,)          = [P*G] # 2 rewards per group
    advantages                : (2, 121)      = [B, T]  (broadcast from per-rollout scalar)
    old_logps / ref_logps     : (2, 121)      = [B, T] # no grad, same log-prob size
    current_logps             : (2, 121)      = [B, T], requires_grad=True
    loss                      : scalar (), requires_grad=True

    1 prompt, 2 samples per batch means B = 2
    This run's max sequence was of length 122, so L = 122 and thus L - 1 = 121
    """
    main()
