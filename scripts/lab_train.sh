#!/usr/bin/env bash
# Train on a lab box by first staging the processed chunks to node-local disk.
#
# Why: the processed graph chunks (~24 GB across 73 files) live on the shared
# NFS server. The streaming data loader re-reads every chunk once per epoch, so
# on the lab boxes the training loop spends most of its time blocked in
# uninterruptible I/O wait (state `D`, `wait_on_page_bit_common`) reading from
# NFS — it looks frozen. Copying the chunks once to fast local disk (/tmp, the
# lab boxes have ~90 GB free on /) and pointing training at that copy via
# CERTGNN_PROCESSED_DIR makes every epoch read from local SSD instead.
#
# Usage (e.g. from the job queue):
#   bash scripts/lab_train.sh --task anomaly_aware --model gcn_lstm --max-epochs 25
#
# Respects UV_PROJECT_ENVIRONMENT (the queue wrapper sets it to .venv-lab).

set -euo pipefail

cd "$(dirname "$0")/.."

SRC="data/processed/r5.2"
STAGE_ROOT="${CERTGNN_STAGE_ROOT:-/tmp/certgnn-${USER:-x}-$$}"
DST="${STAGE_ROOT}/r5.2"

cleanup() { rm -rf "${STAGE_ROOT}" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "${DST}"
echo "[lab_train] staging chunks: ${SRC} -> ${DST}"
# -n: don't re-copy files already staged (cheap re-runs on the same node).
cp -n "${SRC}"/*.pt "${SRC}"/*.pkl "${DST}/"
echo "[lab_train] staged $(ls "${DST}"/*.pt 2>/dev/null | wc -l) chunk files ($(du -sh "${DST}" | cut -f1))"

export CERTGNN_PROCESSED_DIR="${DST}"
echo "[lab_train] CERTGNN_PROCESSED_DIR=${CERTGNN_PROCESSED_DIR}"

uv run --no-sync train "$@"
