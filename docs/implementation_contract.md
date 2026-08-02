# Implementation Contract

You own `src/grpo/objective.py`. Do not fill it from an LLM without first deriving
the equations and tensor shapes yourself.

## Inputs and shapes

- `logits`: `[B, L, V]`, causal-LM logits for prompt plus completion tokens.
- `input_ids`: `[B, L]`; the probability for target `input_ids[:, t]` comes from
  `logits[:, t - 1]`.
- `completion_mask`: `[B, L - 1]`, where 1 means the corresponding shifted target
  is a generated completion token. Prompt and padding targets are 0.
- `rewards`: `[P, G]`, with `P` prompts and `G` rollouts per prompt.
- `current_logps`, `old_logps`, `ref_logps`, and `completion_mask`: `[P*G, T]`.

## Required behavior

1. `completion_logprobs` returns gathered log-probabilities for shifted target tokens
   and must not include prompt/pad tokens in its later reduction.
2. Advantages are normalized **within each prompt's rollout group** and are then
   broadcast to every valid completion token of that rollout.
3. Old-policy and reference values must not receive gradients. Current-policy
   log-probabilities must retain gradients.
4. All objective terms are masked and normalized by the number of valid completion
   tokens. Empty masks must raise a useful error rather than silently produce NaN.
5. A zero-variance reward group must have defined, finite behavior; choose and document
   it in `decisions.md` before implementing it.

## Design decisions you must make

- population vs. sample standard deviation
- epsilon placement for advantage normalization
- whether and how many PPO epochs are used per rollout batch
- reference-policy cadence
- reward parsing rules and malformed-output behavior

