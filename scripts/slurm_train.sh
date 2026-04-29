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
# - batch_size 1024: model is 53 K params, GPU memory is barely touched at any size
# - num_workers 4: each worker holds its own LRU chunk cache; with auto-detection
#                  we get ~6 chunks per worker (4 x 6 x 1.14 GB ~= 27 GB, fits in 32 GB)
# - prefetch_factor 4: every worker keeps 4 batches in-flight to hide chunk-swap stalls
# - persistent_workers auto-resolves to True on CUDA (skip phase-respawn cost)
# - precision auto-resolves to bf16-mixed on Ampere (RTX 3090 supports it natively)
# - target_fpr 0.05 matches paper section V-C r5.2 operating point;
#   override to 0.09 if you switch to r6.2.

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
# --no-sync: assume `.venv` was prepared by `uv sync` BEFORE sbatch. Two parallel
# jobs would otherwise race on the venv lock. Run `uv sync` manually after each
# pyproject.toml change.
uv run --no-sync python src/certgnn/train.py \
    --processed-dir data/processed/r5.2 \
    --batch-size 1024 \
    --max-epochs 50 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --persistent-workers auto \
    --log-every-n-steps 50 \
    --target-fpr 0.05 \
    --val-size 0.1 \
    --test-size 0.2

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
