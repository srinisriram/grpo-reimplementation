# GRPO Reimplementation

A learning-first scaffold for implementing the algorithmic core of Group Relative
Policy Optimization (GRPO). Infrastructure is provided; the mathematical core is
intentionally unimplemented in `src/grpo/objective.py`.

## Status

Phase 3 complete: baseline GRPO on Qwen2.5-1.5B produces a statistically significant
+14.2 point improvement on GSM8K (52.1% → 66.3%, n=1319 full test set, z=7.41 at the
final checkpoint). See [`docs/phase3_results.md`](docs/phase3_results.md) for the full
analysis, including the trajectory across training and known caveats.

Phase 4 (extension) planning in progress.

## What you implement

- completion-token log probabilities
- group-relative advantages
- importance ratios and clipped surrogate objective
- KL estimator
- final masked GRPO loss and all detach/gradient decisions

The unit tests in `tests/unit/` are the specification for those functions. They
are expected to fail until you implement them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Suggested order

1. Read `docs/implementation_contract.md` and derive each operation by hand.
2. Implement one function at a time in `src/grpo/objective.py`.
3. Run its matching tests, then `pytest tests/unit`.
4. Run `python -m grpo.train --config configs/smoke.yaml` only after the loss is implemented.

## Commands

```bash
pytest tests/unit
pytest tests/integration
python -m grpo.train --config configs/smoke.yaml
python -m grpo.evaluate --config configs/smoke.yaml --checkpoint path/to/adapter
```

`smoke.yaml` is deliberately tiny. Do not treat it as a scientific training configuration.

