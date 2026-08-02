"""Generation and batch-shaping infrastructure; no RL objective math lives here."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """A prompt-major group of generated responses ready for your objective."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    prompt_count: int
    group_size: int
    texts: list[str]


@torch.no_grad()
def generate_groups(model, tokenizer, prompts: list[str], *, group_size: int, max_new_tokens: int) -> RolloutBatch:
    """Sample `group_size` responses per prompt and construct a shifted completion mask.

    The returned rows are prompt-major: rows `[i*G:(i+1)*G]` belong to prompt `i`.
    This ordering is part of the contract for reward grouping and advantages.
    """
    if not prompts:
        raise ValueError("generate_groups requires at least one prompt")
    device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_width = encoded.input_ids.shape[1]
    repeated_ids = encoded.input_ids.repeat_interleave(group_size, dim=0)
    repeated_attention = encoded.attention_mask.repeat_interleave(group_size, dim=0)
    generated = model.generate(
        input_ids=repeated_ids, attention_mask=repeated_attention, do_sample=True,
        num_return_sequences=1, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
    )
    attention_mask = generated.ne(tokenizer.pad_token_id).long()
    # Shifted target t corresponds to token input_ids[:, t + 1].
    positions = torch.arange(generated.shape[1] - 1, device=device).unsqueeze(0)
    completion_mask = (positions >= prompt_width - 1) & attention_mask[:, 1:].bool()
    texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    return RolloutBatch(generated, attention_mask, completion_mask, len(prompts), group_size, texts)

