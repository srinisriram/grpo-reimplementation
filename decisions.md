# Decisions Log

Use this template for every consequential design decision.

- KL direction: log_ratio = ref_logps - current_logps, so s = π_ref/π_θ.
- Importance ratio direction: exp(current_logps - old_logps), so ratio = π_current/π_old.
- Std: population, chosen because matches DeepSeekMath.
- Zero-variance handling: divide by (std + eps).
- Pooled global mean: longer completions get proportionally more gradient influence; accepted deliberately since amplifying the penalty on long, wrong completions is desirable for a math-reasoning model, not just a side effect to tolerate
- `ppo_epochs=1` (default, configurable). Old and current policy are the same weights at the only gradient step this batch, so `ratio ≈ 1` and `clip_range` doesn't engage yet — deliberate, to keep Phase 2 debugging to one axis at a time.
- Reference policy fixed for the entire run — implemented via `model.disable_adapter()` on the same LoRA-wrapped model rather than a separate frozen model copy, since the base weights never change and LoRA is zero-initialized. No duplicate model in memory.
- Checkpointing every `save_steps` (config, defaults to `max_steps` i.e. final-only if unset) plus always on the last step.
