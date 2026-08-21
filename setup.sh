# GRPO reimplementation -- fresh-instance setup.
#
# Must be SOURCED, not executed, so the venv activation and exported env vars
# persist into your actual shell instead of vanishing with a subshell:
#
#   source setup.sh
#
# (`bash setup.sh` or `./setup.sh` will run everything but leave your shell
# without the venv active or the env vars set -- the whole point of sourcing.)
#
# Assumes you're already inside a freshly-cloned copy of the repo.

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Creating virtual environment..."
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "==> Installing project + dev dependencies..."
pip install --upgrade pip -q
pip install -e '.[dev]' -q

# Found on a previous CUDA rig (not seen on MPS): newer transformers can
# default to compiled generation, which stalled indefinitely on the first
# generate() call with no error, just zero progress. Blunt but effective
# workaround -- see decisions.md for the full story.
export TORCHDYNAMO_DISABLE=1

# Deliberately NOT setting CUDA_VISIBLE_DEVICES here. On the old rig, GPU 0
# had to be excluded because it was also driving a display -- that doesn't
# apply to a rented headless instance, where every GPU should be fair game.
# If this specific rental does reserve a GPU for something else, set
# CUDA_VISIBLE_DEVICES manually before training.

if [ -n "${HF_TOKEN:-}" ]; then
    echo "==> HF_TOKEN is set, will be used for Hugging Face Hub requests."
else
    echo "==> No HF_TOKEN set -- downloads will work but may be rate-limited."
    echo "    export HF_TOKEN=... before sourcing this script if you have one."
fi

echo "==> Running unit tests as a sanity check on the environment..."
pytest tests/unit -q

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo ""
echo "==> Setup complete. Detected ${GPU_COUNT} GPU(s). Venv is active in this shell."
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "    accelerate launch --num_processes=${GPU_COUNT} --multi_gpu -m grpo.train --config configs/full_run.yaml"
else
    echo "    python -m grpo.train --config configs/full_run.yaml"
fi
