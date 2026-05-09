#!/usr/bin/env bash
#SBATCH --job-name=gnn-insider
#SBATCH --partition=student-nvidia
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output="/Ziob/awalczak/gnn-insider-threat/logs/train-%j.txt"
#SBATCH --error="/Ziob/awalczak/gnn-insider-threat/logs/train-%j.txt"

# RTX 3090 (24 GB VRAM) + 32 GB system RAM + 8 CPUs.
# Hardware-specific knobs (batch_size, num_workers, prefetch_factor,
# persistent_workers, precision) are now expressed in configs/config.yaml
# under training.data and training.trainer. This script only overrides
# the few flags worth keeping per-job (max_epochs, model/task selection)
# via train's CLI flags; everything else flows through config.

set -euo pipefail

# --- Paths and env ---
PROJECT_DIR="/Ziob/awalczak/gnn-insider-threat"
cd "$PROJECT_DIR"
mkdir -p logs

# uv must be on PATH; SLURM doesn't source .bashrc.
export PATH="$HOME/.local/bin:$PATH"

# uv config: copy-mode link is needed when cache and venv live on different
# filesystems (cluster home vs /Ziob), where hardlinks fail across mounts.
export UV_CACHE_DIR="/Ziob/awalczak/.cache/uv"
export UV_LINK_MODE=copy

# Keep ML caches off $HOME — it's a quota-limited NFS on the student cluster.
export HF_HOME="/Ziob/awalczak/.cache/huggingface"
export TORCH_HOME="/Ziob/awalczak/.cache/torch"

# --- Diagnostics (visible in train-${jobid}.txt for post-mortem) ---
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID:-?}"
echo "Node:      $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Workdir:   $(pwd)"
echo "Start:     $(date)"
echo "=========================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "=========================================="

# --- Training ---
# --no-sync: assume `.venv` was prepared by `uv sync` BEFORE sbatch. Two
# parallel jobs would otherwise race on the venv lock. Run `uv sync` manually
# after each pyproject.toml change.
#
# Forward any extra args to `uv run train`, e.g.:
#   sbatch scripts/slurm_train.sh --task binary --model graph_pool_mlp
uv run --no-sync train "$@"

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
