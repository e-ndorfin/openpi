#!/bin/bash
# Submit from the OpenPI repository with:
#   sbatch scripts/train_wire_insertion_success.slurm
#
# Optional overrides:
#   EXP_NAME=my_run CHECKPOINT_BASE_DIR=/data1/zach/openpi/checkpoints \
#     FSDP_DEVICES=1 NUM_WORKERS=16 sbatch scripts/train_wire_insertion_success.slurm
#   SMOKE_TEST=1 sbatch scripts/train_wire_insertion_success.slurm

#SBATCH --job-name=pi05-wire-success
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data1/zach/openpi/slurm-%x-%j.out
#SBATCH --error=/data1/zach/openpi/slurm-%x-%j.err
#SBATCH --requeue

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/data1/zach/openpi}}"
CONFIG="pi05_wire_insertion_success"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints}"
WIRE_INSERTION_DATA_ROOT="${WIRE_INSERTION_DATA_ROOT:-/data1/zach/lerobot_data/wire_insertion}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NORM_STATS="$REPO_ROOT/assets/$CONFIG/wire_insertion/norm_stats.json"

# Use the stable Slurm submission time rather than the process start time. A
# requeued job therefore keeps the same experiment directory and resumes, while
# every independent sbatch submission gets a new timestamped directory.
SUBMIT_TIME="$(squeue --noheader --job="${SLURM_JOB_ID:?}" --format=%V)"
RUN_TIMESTAMP="$(date --date="$SUBMIT_TIME" +%Y%m%d-%H%M%S)"
EXP_NAME="${EXP_NAME:-pi05_wire_insertion_success_$RUN_TIMESTAMP}"

cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: REPO_ROOT is not an OpenPI checkout: $REPO_ROOT" >&2
    exit 1
fi
if [[ ! -d "$WIRE_INSERTION_DATA_ROOT/data" || ! -f "$WIRE_INSERTION_DATA_ROOT/meta/info.json" ]]; then
    echo "ERROR: invalid wire-insertion dataset root: $WIRE_INSERTION_DATA_ROOT" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: missing OpenPI environment. Run 'GIT_LFS_SKIP_SMUDGE=1 uv sync' first." >&2
    exit 1
fi

export WIRE_INSERTION_DATA_ROOT
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"

echo "node:             $(hostname)"
echo "CUDA devices:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "config:           $CONFIG"
echo "experiment:       $EXP_NAME"
echo "dataset:          $WIRE_INSERTION_DATA_ROOT"
echo "checkpoints:      $CHECKPOINT_BASE_DIR"
echo "FSDP devices:     $FSDP_DEVICES"
echo "data workers:     $NUM_WORKERS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    echo "Running one-batch data-pipeline smoke test; training will not start."
    uv run --locked python - <<'PY'
import dataclasses

from openpi.training import config as config_module
from openpi.training import data_loader

config = dataclasses.replace(
    config_module.get_config("pi05_wire_insertion_success"),
    batch_size=1,
    num_workers=0,
)
loader = data_loader.create_data_loader(
    config,
    skip_norm_stats=True,
    num_batches=1,
    shuffle=False,
)
observation, actions = next(iter(loader))
print("SMOKE TEST PASSED")
print(f"actions={actions.shape} state={observation.state.shape}")
print(f"images={{{', '.join(f'{key}: {value.shape}' for key, value in observation.images.items())}}}")
print(f"tokens={observation.tokenized_prompt.shape}")
PY
    exit 0
fi

if [[ ! -f "$NORM_STATS" ]]; then
    echo "Normalization stats not found; computing them at $NORM_STATS"
    uv run --locked scripts/compute_norm_stats.py --config-name="$CONFIG"
else
    echo "Using existing normalization stats: $NORM_STATS"
fi

# --resume is safe for a new run and lets a requeued or resubmitted job continue
# from the most recent checkpoint without deleting prior work.
uv run --locked scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --num-workers="$NUM_WORKERS" \
    --fsdp-devices="$FSDP_DEVICES" \
    --resume
