from __future__ import annotations

import torch


def completion_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) \
    -> torch.Tensor:
    """
    Returns a log-probabilities tensor of shape [B, L-1] where B is the batch 
    size per step (in the real pipeline this is P prompts * G samples per 
    prompt, each batch contains B sequences) and L is the max sequence length 
    of the batch. Each entry in the returned tensor is the log-probability of 
    the actual selected token from input_ids extracted from the model's current 
    log-probabilities at the token t. Mathematically, each entry is 
    log π(token_t | tokens_<t) for a particular token t in a batch sequence.
    --
    Parameters:
        logits: model's logits as a [B, L, V] tensor where V is the size
                of the model's vocab
        input_ids: the actual token chosen in the sequence as a [B, L] tensor
    --
    Returns:
        The log-probabilities tensor of each token in a sequence as a [B, L-1]
        tensor.
    """
    log_prob_logits = torch.log_softmax(logits, dim=-1)
    
    pred_probs = log_prob_logits[:, :-1, :]
    
    actual_next = input_ids[:, 1:]
    
    return torch.gather(pred_probs, -1, actual_next.unsqueeze(-1)).squeeze(-1)


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-4) \
    -> torch.Tensor:
    """
    Returns a standardized rewards tensor of shape [num_prompts, group_size]
    where each entry in the array is the standardized reward score of that
    particular response with respect to the other responses in its group.
    Standardizes with the z-score formula, including a small eps value in the 
    std denominator to prevent blow-up in cases where all reward scores are
    the same within a group. Use population std (unbiased=False) for std
    calculation to match DeepSeekMath's implementation.
    --
    Parameters:
        rewards: rewards of each response in a group as a 
                 [num_prompts, group_size] tensor.
        eps: float epsilon value to prevent blow-up, default = 1e-4
    --
    Returns:
        Normalized reward scores per-group as a [num_prompts, group_size]
        tensor, returned as advantages tensor everywhere downstream.
    """
    mean_values = rewards.mean(dim=-1, keepdim=True)
    
    std_values = rewards.std(dim=-1, unbiased=False, keepdim=True)
    
    return (rewards - mean_values) / (std_values + eps)
    
def importance_ratio(current_logps: torch.Tensor, old_logps: torch.Tensor) \
    -> torch.Tensor:
    """
    To measure the correction factor of how much the current log probabilities
    have drifted from the old log probabilities per-token we use the formula
    e^{current_logps - old_logps} which is equivalent to \frac{π_current}
    {π_old}. This is fed into the clipped surrogate loss to ensure that
    we are correcting for the fact that rollouts are sampled from an old model,
    and to ensure that we do not move too far outside the trust region
    around the old model's probabilities.
    --
    Parameters:
        current_logps: output of completion_logprobs for the current model's
                       log-probabilities as a [B, L-1] tensor
        old_logps: output of completion_logprobs for the old model's
                   log-probabilities as a [B, L-1] tensor
    --
    Returns:
        Per-token ratio of the actual probabilities of current vs. old model.
    """
    log_ratio = current_logps - old_logps
    
    return torch.exp(log_ratio)


def kl_estimator(current_logps: torch.Tensor, ref_logps: torch.Tensor) \
    -> torch.Tensor:
    """
    To estimate kl drift from the reference model, frozen at its state from the 
    start of training, we use the stable, unbiased, and non-negative formula
    s - r - 1 per-token, where s is the ratio of the actual probabilities and 
    r is the difference in log-probabilities (ref_logps - current_logps) to 
    match DeepSeekMath's implementation and to measure the correct divergence. 
    --
    Parameters:
        current_logps: output of completion_logprobs for the current model's
                       log-probabilities as a [B, L-1] tensor
        ref_logps: output of completion_logprobs for the reference model's
                   log-probabilities as a [B, L-1] tensor
    --
    Returns:
        The KL penalty per-token as a [B, L-1] tensor.
    """
    log_ratio = ref_logps - current_logps
    
    s = torch.exp(log_ratio)
    
    kl = s - log_ratio - 1
    
    return kl


def grpo_loss(
    current_logps: torch.Tensor, old_logps: torch.Tensor, ref_logps: torch.Tensor,
    advantages: torch.Tensor, completion_mask: torch.Tensor, clip_range: float, 
    kl_coef: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Returns the GRPO loss as a scalar for the entire training batch. Per the
    DeepSeekMath objective, the loss is the negative of the objective (since
    maximizing the objective is the same as minimizing the negative objective),
    which is: -min(ratio_t * A_t, clip(ratio_t, 1-ε, 1+ε) * A_t) + β * KL_t.
    We take the average of all the individual losses per token across all 
    valid objective tokens (non-padding or prompt tokens). Returns a tuple
    of the scalar loss for this batch along with a diagnostic dictionary.
    --
    Parameters:
        current_logps: output of completion_logprobs for the current model's
                       log-probabilities as a [B, L-1] tensor
        old_logps: output of completion_logprobs for the old sampling model's
                   log-probabilities as a [B, L-1] tensor
        ref_logps: output of completion_logprobs for the reference model's
                   log-probabilities as a [B, L-1] tensor
        advantages: output of group_relative_advantages function, broadcasted 
                    per-token as a [B, L-1] tensor
        completion_mask: 0-1 tensor of size [B, L-1], 0 if the token is padding
                         or prompt, 1 if the token is an actual generated token
                         by the model.
        clip_range: epsilon value used in DeepSeekMath's clipped surrogate loss
                    function, as a float
        kl_coef: the value of β in the DeepSeekMath objective, as a float
    --
    Returns:
        A tuple of the form (torch.Tensor, dict[str, torch.Tensor]). First value
        is the scalar loss for this batch as a scalar tensor, and the second
        value is a diagnostic dictionary of the pure-policy loss without the
        KL penalty, the raw KL mean estimate, and the 
        clip fraction (proportion of times the clipped term was selected in the 
        minimum surrogate).
    """
    if completion_mask.sum() == 0:
        raise ValueError("completion mask does not contain any valid objective \
            tokens")
        
    ratio = importance_ratio(current_logps, old_logps)
    
    clip_ratio = torch.clamp(ratio, min=1-clip_range, max=1+clip_range)
    
    L_clip = torch.min(ratio * advantages, clip_ratio * advantages)
    
    kl_per_token = kl_estimator(current_logps, ref_logps)
    
    # decision: use global pooled mean instead of per-sequence to provide 
    # more gradient influence for longer completions
    
    clip_mean = (L_clip * completion_mask).sum() / (completion_mask.sum())
    
    kl_mean = (kl_per_token * completion_mask).sum() / (completion_mask.sum())
    
    loss = -clip_mean + kl_coef * kl_mean
    
    clip_fraction = (((L_clip == clip_ratio * advantages) & 
                      ((ratio < 1-clip_range) | (ratio > 1+clip_range)) & 
                      completion_mask.bool())).sum() / (completion_mask.sum())
    
    return (loss, {"policy_loss": -clip_mean.detach(), "kl": kl_mean.detach(), 
                   "clip_fraction": clip_fraction})


# Deliberately no partial implementations live here. A mathematically plausible
# default is worse than an explicit gap in a learning-first project: it makes it
# too easy to train against code whose reduction, masking, or detach semantics
# you have not personally checked.