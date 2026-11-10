#!/usr/bin/env bash
# Run from any directory. All stages use physical GPU 3 by default.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${COMMU_GPU_INDEX:-3}"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export TMPDIR="$PROJECT_DIR/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PROJECT_DIR/.venv/bin:$PROJECT_DIR/.tools/bin:$PATH"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project environment: $PYTHON" >&2
  exit 1
fi
STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "Usage: bash runpod/lab_speech_to_motion.sh {check|prepare|features|frozen|finetune|tensorboard|module} [arguments]" >&2
  exit 1
fi
shift
case "$STAGE" in
  check)
    "$PYTHON" -c 'import torch, transformers; print("Torch:", torch.__version__); print("Transformers:", transformers.__version__); print("GPU:", torch.cuda.get_device_name(0)); print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30)); print("BF16:", torch.cuda.is_bf16_supported()); assert torch.cuda.device_count() == 1'
    exec "$PYTHON" -m pytest speech_to_motion/tests -q "$@"
    ;;
  prepare) exec "$PYTHON" -m speech_to_motion.prepare_dataset "$@" ;;
  tensorboard)
    "$PYTHON" -m speech_to_motion.tensorboard_logging
    exec "$PYTHON" -m tensorboard.main --logdir outputs/speech_to_motion \
      --host 127.0.0.1 --port 6006 --reload_interval 5 "$@"
    ;;
  features) exec "$PYTHON" -m speech_to_motion.extract_features --device cuda --batch-size 16 "$@" ;;
  frozen) exec "$PYTHON" -m speech_to_motion.train --stage frozen --run-name base_all_frozen "$@" ;;
  finetune)
    exec "$PYTHON" -m speech_to_motion.train --stage finetune --run-name base_all_last2 \
      --init-checkpoint outputs/speech_to_motion/base_all_frozen/checkpoints/last.pt \
      --batch-size 8 --accumulation-steps 4 --unfreeze-last-n 2 "$@"
    ;;
  module) exec "$PYTHON" -m "$@" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
