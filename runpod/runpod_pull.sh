#!/usr/bin/env bash
#
# Pull checkpoints (and TensorBoard run logs) back from a RunPod instance.
#
# This is the complement to runpod_dinov3.sh: after training finishes on the
# pod, run this to copy the trained checkpoints and TensorBoard logs back to
# your local machine.
#
# Usage:
#   ./runpod/runpod_pull.sh
#
# Edit the connection variables below to match your RunPod instance.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Connection settings (edit these to match runpod_dinov3.sh)
# --------------------------------------------------------------------------- #
RUNPOD_HOST="213.173.102.146"
RUNPOD_PORT="39323"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/minhle"

# Remote source directory (must match REMOTE_DIR in runpod_dinov3.sh).
REMOTE_DIR="/workspace/commu-voice_2_action"

# --------------------------------------------------------------------------- #
# Local destination paths (relative to the repo root)
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_CKPT="$REPO_ROOT/pose_estimation/checkpoints"
LOCAL_RUN="$REPO_ROOT/pose_estimation/run"

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
mkdir -p "$LOCAL_CKPT" "$LOCAL_RUN"

echo "==> Pulling checkpoints from $RUNPOD_HOST"
rsync -avz -e "ssh -p $RUNPOD_PORT -i $SSH_KEY" \
    --exclude '._*' \
    --exclude '.DS_Store' \
    "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/pose_estimation/checkpoints/" \
    "$LOCAL_CKPT/"

echo "==> Pulling TensorBoard run logs from $RUNPOD_HOST"
rsync -avz -e "ssh -p $RUNPOD_PORT -i $SSH_KEY" \
    --exclude '._*' \
    --exclude '.DS_Store' \
    "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/pose_estimation/run/" \
    "$LOCAL_RUN/"

echo "==> Done. Checkpoints -> $LOCAL_CKPT"
echo "==> Done. Run logs     -> $LOCAL_RUN"
