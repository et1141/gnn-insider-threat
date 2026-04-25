#!/usr/bin/env bash
# Wrapper that starts training with allocator settings tuned for Apple Silicon.
#
# MallocNanoZone=0   — disables nano malloc. The default allocator on Apple
#                      Silicon never returns freed pages to the OS, so a
#                      long-running PyTorch process accumulates tens of GB of
#                      fragmented heap. The standard magazine allocator does
#                      release.
# PYTHONMALLOC=malloc — bypass pymalloc's per-arena pool (it keeps a 256 KB
#                      arena alive as long as one object inside is live, which
#                      causes massive fragmentation under repeated torch.load).
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 — push MPS to release Metal pool memory
#                      eagerly instead of holding multiple GB indefinitely.
#
# Both Malloc* vars must be set BEFORE the Python interpreter starts; setting
# them inside Python is too late.
#
# Usage: ./scripts/train.sh [args forwarded to src/certgnn/train.py]

set -euo pipefail

cd "$(dirname "$0")/.."

export MallocNanoZone=0
export PYTHONMALLOC=malloc
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

exec uv run python src/certgnn/train.py "$@"
