# Phase 3: Baseline GRPO Results

## Configuration

- Model: Qwen/Qwen2.5-1.5B-Instruct
- Dataset: GSM8K (main config)
- LoRA rank 8, alpha 16, target modules [q_proj, v_proj]
- 600 steps, per_device_batch_size 2, group_size 4
- Learning rate: 2.0e-5 → 5.0e-6, cosine schedule (5-step linear warmup, then cosine decay to the floor)
- KL coef 0.04, clip range 0.2
- Save every 50 steps (12 checkpoints: 50, 100, ..., 600)
- Seed 7
- max_new_tokens 512
- Gradient checkpointing enabled (activation memory, not the memory bottleneck at this model size otherwise)
- Config: `configs/full_run_600.yaml`

## Hardware

- Single RTX 4080 Super, 16GB VRAM (per user)
- Single-process (`python -m grpo.train`, no `accelerate --multi_gpu`) — this rig has one GPU

## Results (n=1319, full GSM8K test set)

```
checkpoint        accuracy   correct  mean_len  correct_len  wrong_len
----------------------------------------------------------------------
base                 0.521   687/1319     334.4        298.3      373.7
checkpoint-50        0.541   713/1319     328.9        295.5      368.3
checkpoint-100       0.587   774/1319     309.1        280.5      349.8
checkpoint-150       0.613   808/1319     305.2        278.3      347.7
checkpoint-200       0.606   799/1319     305.7        274.3      354.0
checkpoint-250       0.642   847/1319     297.5        273.4      340.7
checkpoint-300       0.638   841/1319     300.7        276.3      343.6
checkpoint-350       0.649   856/1319     295.4        269.1      343.9
checkpoint-400       0.650   858/1319     296.1        271.6      341.6
checkpoint-450       0.654   863/1319     292.7        270.4      335.0
checkpoint-500       0.657   867/1319     296.0        274.3      337.7
checkpoint-550       0.659   869/1319     299.0        276.3      342.8
checkpoint-600       0.663   874/1319     295.9        274.5      338.1
```

## Statistical Analysis

Two-proportion z-test, each checkpoint vs. base, `n=1319`:

- Every checkpoint from `checkpoint-100` onward is significant vs. base at `p<0.05` (single comparison): `z` ranges from `3.41` (checkpoint-100) to `7.41` (checkpoint-600).
- `checkpoint-50` is not significant (`z=1.01`) — consistent with the trajectory showing training hadn't produced a real effect yet that early.
- 95% CI half-width at `n=1319`, `p≈0.52-0.66`: approximately `±2.7` points.
- Absolute improvement, base → checkpoint-600: `52.1% → 66.3%`, `+14.2` points.

Caveat: these are naive single-comparison z-tests, not corrected for testing 12 checkpoints against the same base, and not accounting for the fact that checkpoints share the same eval set (correlated errors) or come from one continuous training trajectory rather than independent runs. Given the effect sizes and z-scores involved here, a more conservative/paired test would very likely still show significance from checkpoint-100 onward, but that hasn't been formally run.

## Trajectory Analysis

Marginal accuracy gain per 50-step window (percentage points, relative to the previous checkpoint):

```
checkpoint-50:  +1.97
checkpoint-100: +4.62
checkpoint-150: +2.58
checkpoint-200: -0.68
checkpoint-250: +3.64
checkpoint-300: -0.45
checkpoint-350: +1.14
checkpoint-400: +0.15
checkpoint-450: +0.38
checkpoint-500: +0.30
checkpoint-550: +0.15
checkpoint-600: +0.38
```

- Improvement is monotonic-ish through ~step 350 (two small non-monotonic dips at 200 and 300, but the overall trend through this range is clearly upward).
- Clear plateau from ~step 350 through 600 — marginal gain per 50-step window drops to `+0.15` to `+0.38`, versus `+4.62`/`+3.64` in the early-to-mid run. 150 additional steps (450→600) bought well under one accuracy point.
- This is a genuinely useful trajectory shape for future runs: most of the accuracy is captured well before 600 steps, though the exact plateau point (and whether it holds for a different model size or reward setup) hasn't been separately tested.

## Key Findings

1. GRPO produces a statistically significant improvement on Qwen2.5-1.5B / GSM8K under this configuration — the first run in this project's history to clear a rigorous significance bar (prior 100-step and 300-step runs, evaluated at smaller `n`, did not).
2. The effect saturates around `checkpoint-450`–`600` (~65-66% accuracy) — diminishing returns are clear and quantified above.
3. Completion length dropped (`334.4 → 295.9` tokens, base to checkpoint-600) while accuracy rose. This is evidence *against* the specific failure mode worried about earlier (pooled global mean disproportionately suppressing hard-problem learning via length-based punishment) — the correct/wrong length gap narrowed, not widened, over training. Not proof the pooled-mean design is optimal, but no evidence it actively hurt here.
4. KL divergence stayed bounded and small throughout (`0.001`–`0.004` by the end of training) — no sign of instability or runaway drift from the reference policy.
5. A self-contradiction failure mode was identified via manual completion inspection during the 100-step run's diagnostics (model derives a correct answer in its reasoning, then overrides it with a different, wrong final `\boxed{}` answer) and confirmed to still occur after training — the current reward function (`gsm8k_reward`, which scores the *last* `\boxed{}` in a completion) has no specific mechanism to discourage this behavior. Not addressed in this run; a candidate fix (score the first `\boxed{}` instead of the last) was proposed but not implemented or tested.

## Methodology Notes

- Earlier runs (100 steps, 300 steps), evaluated at smaller `n` (100 and 300 respectively), showed wider confidence intervals (`±9.8` and `±5.7` points) that made whatever real signal existed harder to distinguish from noise. `n=1319` (the full GSM8K test set) was what it took to detect the effect with a clean single-comparison significance result.
- The 100-step and 300-step runs used a **linearly decaying LR** (warmup then linear decay to exactly 0 by the final step, via `get_linear_schedule_with_warmup`) — not a constant LR, as an earlier draft of this document incorrectly stated. This run used a **cosine schedule decaying to a nonzero floor** (`5e-6`) instead. The comparison across runs is therefore "linear-to-zero vs. cosine-to-a-floor," not "constant vs. cosine" — a real difference, but a more specific one. The cosine schedule's own contribution relative to the linear one is not isolated (would need a controlled same-step-count comparison, not run here).
- An earlier hypothesis — that the 100-step run's checkpoint-100 was "already showing the real effect, just undetectable due to noise" — should be treated as plausible, not established. The LR schedule shape differs between runs from the first real gradient update onward (different `num_training_steps` changes the decay curve immediately, even before eval-detectability differences), so checkpoints at the same step count across different-length runs are not directly the same experiment paused at different points. The available data (100-step run's checkpoint-100: `54.0%` at `n=300`, CI roughly `44-60%`; this run's checkpoint-100: `58.7%` at `n=1319`) is *consistent with* but does not *confirm* that claim.

## Diagnostic Findings (from earlier runs, still relevant)

- Truncation: not a significant bottleneck. In the 100-step run's diagnostic, only ~1% of examples showed the strongest truncation signature (hit the token cap *and* never produced a `\boxed{}` answer at all) — the model states its boxed answer within the first few tokens of a completion, so running out of budget almost never costs it the scored answer.
- Length-vs-correctness gap: present in the base model (wrong completions longer than correct ones) and narrowed, not widened, after training — see Key Finding 3 above.
- Self-contradiction pattern: confirmed on the same GSM8K problem in both the base model and a trained checkpoint from the 100-step run — see Key Finding 5.

## Files

- Training config: `configs/full_run_600.yaml`
- Training log: not saved to a file (all runs printed to an interactive `screen` session with no redirection); see project conversation history for excerpts
- Eval results: full table above, generated via `python scripts/compare_checkpoints.py --config configs/full_run_600.yaml --num_examples 1319 --dump_dir outputs/full_run_600/dumps`
- Checkpoints: `outputs/full_run_600/checkpoint-{50,100,...,600}/` (LoRA adapters only, ~4.2MB each)
- Completion dumps: `outputs/full_run_600/dumps/completions_{base,checkpoint-*}.jsonl` (one JSON record per eval example: question, completion, predicted/reference answer, length, hit_token_cap, correct)
- Analysis reports: not yet generated for this run's dumps (`scripts/analyze_completions.py` was run against the 100-step run's dumps only so far)

## Next Steps

Phase 4 direction TBD. Candidates:
- Consistency-rewarded GRPO (address the self-contradiction failure mode directly, e.g. by scoring the first `\boxed{}` rather than the last)
- Scale to Qwen2.5-3B or 7B
- WS-GRPO variant inspired by Julian McAuley's paper
- Comparison of reward function variants
