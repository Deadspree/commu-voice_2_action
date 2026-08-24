#!/usr/bin/env bash
#
# Copy the files needed for training to a RunPod instance via scp.
#
# Copies:
#   - dataset/commu_pose_dataset   (training data)
#   - pose_estimation/             (training code + weights + checkpoints)
#   - requirements.txt             (Python dependencies)
#   - README.md                    (project docs)
#
# The DINOv3 backbone (third_party/dinov3) is NOT copied; it is a git repo and
# is cloned directly on the pod (see the clone step at the end).
#
# Usage:
#   ./runpod/runpod_dinov3.sh
#
# Edit the connection variables below to match your RunPod instance.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Connection settings (edit these)
# --------------------------------------------------------------------------- #
RUNPOD_HOST="194.68.245.95"
RUNPOD_PORT="22078"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/minhle"

# Remote destination directory (created if missing).
REMOTE_DIR="/workspace/commu-voice_2_action"

# --------------------------------------------------------------------------- #
# Local source paths (relative to the repo root)
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$REPO_ROOT/dataset/commu_pose_dataset"
POSE_DIR="$REPO_ROOT/pose_estimation"
REQUIREMENTS="$REPO_ROOT/requirements.txt"
README="$REPO_ROOT/README.md"

# --------------------------------------------------------------------------- #
# scp / rsync helpers
# --------------------------------------------------------------------------- #
scp_to() {
    # $1 = local path, $2 = remote destination path
    echo "==> Copying $1 -> $2"
    scp -P "$RUNPOD_PORT" -i "$SSH_KEY" -r "$1" "$RUNPOD_USER@$RUNPOD_HOST:$2"
}

rsync_to() {
    # $1 = local path (trailing slash copies contents), $2 = remote destination path
    echo "==> Copying $1 -> $2"
    # --exclude '._*' skips macOS AppleDouble metadata files, which otherwise
    # cause rsync to fail with exit status 23 (partial transfer).
    rsync -avz -e "ssh -p $RUNPOD_PORT -i $SSH_KEY" \
        --exclude 'checkpoints/*.pt' \
        --exclude '._*' \
        --exclude '.DS_Store' \
        "$1" "$RUNPOD_USER@$RUNPOD_HOST:$2"
}

# Transfer a directory as a single compressed tarball. Much faster than
# copying thousands of small files individually.
#
# macOS bsdtar embeds AppleDouble (._*) files and xattrs by default, which the
# remote GNU tar chokes on. We exclude macOS metadata and tell the remote tar
# to skip ownership/xattr restoration.
tar_to() {
    # $1 = local directory to archive, $2 = remote destination directory
    local src="$1"
    local dest="$2"
    local name
    name="$(basename "$src")"
    echo "==> Archiving + copying $src -> $dest (as $name.tar.gz)"
    tar -C "$(dirname "$src")" --exclude='._*' --exclude='.DS_Store' -czf - "$name" \
        | ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "$RUNPOD_USER@$RUNPOD_HOST" \
            "mkdir -p '$dest' && tar --no-same-owner --no-xattrs -xzf - -C '$dest'"
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
echo "Creating remote directory: $REMOTE_DIR"
# scp does not create intermediate directories, so create all destination
# subdirectories up front.
ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "$RUNPOD_USER@$RUNPOD_HOST" \
    "mkdir -p '$REMOTE_DIR/dataset' '$REMOTE_DIR/pose_estimation'"

# Copy the dataset (images + labels + metadata) as a single compressed tarball.
tar_to "$DATASET_DIR" "$REMOTE_DIR/dataset"

# Copy requirements.txt first (small, quick) so it's present even if a later
# step is slow.
scp_to "$REQUIREMENTS" "$REMOTE_DIR/requirements.txt"

# Copy README.md.
scp_to "$README" "$REMOTE_DIR/README.md"

# Copy the pose_estimation code (includes weights/; excludes checkpoints/*.pt).
rsync_to "$POSE_DIR/" "$REMOTE_DIR/pose_estimation/"

# Clone the DINOv3 backbone directly on the pod (it's a git repo, so cloning
# is cleaner and faster than copying).
echo "==> Cloning DINOv3 on the pod"
ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "$RUNPOD_USER@$RUNPOD_HOST" \
    "mkdir -p '$REMOTE_DIR/third_party' && git clone https://github.com/facebookresearch/dinov3.git '$REMOTE_DIR/third_party/dinov3'"

echo "Done. Files copied to $RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR"
