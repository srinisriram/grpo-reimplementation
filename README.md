# GRPO Reimplementation

A from-scratch implementation of Group Relative Policy Optimization (GRPO),
built without TRL or any RL library. The full algorithmic core — completion
log-probabilities, group-relative advantages, importance ratios, the clipped
surrogate objective, the KL estimator, and all masking and detach semantics —
is implemented in `src/grpo/objective.py` and derived by hand from the
DeepSeekMath formulation.

## Status

Phase 3 complete: baseline GRPO on Qwen2.5-1.5B produces a statistically significant
+14.2 point improvement on GSM8K (52.1% → 66.3%, n=1319 full test set, z=7.41 at the
final checkpoint). See [`docs/phase3_results.md`](docs/phase3_results.md) for the full
analysis, including the trajectory across training and known caveats.

Phase 4 (extension) planning in progress.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Commands

```bash
pytest tests/unit
pytest tests/integration
python -m grpo.train --config configs/smoke.yaml
python -m grpo.evaluate --config configs/smoke.yaml --checkpoint path/to/adapter
```

`smoke.yaml` is deliberately tiny. Do not treat it as a scientific training configuration.

