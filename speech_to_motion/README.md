# Speech → CommU motion

The current default uses **all 593 usable Lucy clips for training**, with no
validation or test partition. The model combines pretrained HuBERT with a
four-layer Transformer motion decoder (width 512, eight heads, feed-forward
width 2048). Train the decoder with frozen HuBERT first, then fine-tune its last
two Transformer layers together with the decoder.

This is offline, bidirectional prediction. Targets are pose-estimator predictions
in degrees. Training metrics measure fit to these targets; there is no held-out
estimate of generalization in the default configuration.

## Setup and configuration

Use the existing project environment and root `requirements.txt`. The added
packages are Transformers 4.x and PyYAML; TensorBoard and matplotlib are also
required for logging. FFmpeg/ffprobe must be available on PATH or configured in
`data.ffmpeg`/`data.ffprobe`. The first encoder load downloads HuBERT weights.
No Lucy API calls are made.

The default `speech_to_motion/configs/baseline.yaml` contains:

```yaml
data:
  prepared_dir: dataset/speech_to_motion_all
  normalization: joint_limits
  training_only: true
  ratios: [1.0, 0.0, 0.0]
```

Paths are relative to the repository root and can be overridden by CLI flags.
Source audio and joint CSVs remain in their existing clip folders under
`dataset/lucy/generation_runs/speech2gesture`. Source provenance is in
`dataset/lucy/speech2gesture/manifest.json`. No clip IDs or machine paths are
embedded in Python. Run from the repository root.

## Prepare all clips

For an existing min-max dataset, combine its manifests without changing its
normalization or recomputing existing features:

```bash
python -m speech_to_motion.migrate_normalization \
  --source dataset/speech_to_motion_minmax \
  --destination dataset/speech_to_motion_all --training-only
```

The new train manifest contains the union of the previous three partitions.
Validation/test manifests are empty. Physical-limit normalization is preserved
exactly. Immutable cached features are hard-linked (copied if necessary), with
an independent registry and dataset identity. Old datasets/runs are preserved.

For a fresh dataset, use `python -m speech_to_motion.prepare_dataset` instead.
Preparation checks completed generation status, timeline origin, audio stream,
joint units/order, finite rows, consecutive frames and increasing timestamps.
Audio and motion are truncated to their shared support. Rejected clips and
reasons are recorded in `summary.json`. With training-only mode enabled, no
source grouping or duration splitting is performed: every usable clip trains.

Targets use the same normalization as `pose_estimation`:

```text
normalized = 2 * (angle - joint_min) / (joint_max - joint_min) - 1
angle = normalized * (joint_max - joint_min) / 2 + (joint_max + joint_min) / 2
```

Physical endpoints map to -1/+1. Raw labels outside their limits are not clamped,
so they can fall outside that interval. Predictions are also unbounded. Legacy
standardized checkpoints retain their original mean/std scaling during inference.

## Extract frozen features

```bash
python -m speech_to_motion.extract_features --device cuda --batch-size 16
```

Windows are four seconds, with two-second training stride. Short clips retain
real duration; the final window covers the tail. Targets are interpolated onto
30 Hz timestamps. Audio is mono 16 kHz and normalized independently per window.
Equal-length windows are batched without padding HuBERT audio. Motion padding
is masked in attention, losses and metrics. Feature timestamps use convolutional
receptive-field centers rather than assuming matching video/audio frame rates.

Extraction fills only missing windows. Former validation/test clips may need
additional windows because training uses a shorter stride; existing features
are reused. Migration marks its registry incomplete until extraction has checked
all windows, preventing training from failing halfway through on a missing cache.
Training-only extraction never evaluates or constructs a validation/test dataset.

## Frozen training and fine-tuning

```bash
python -m speech_to_motion.train --stage frozen --run-name base_all_frozen

# After frozen training finishes:
python -m speech_to_motion.train \
  --stage finetune --run-name base_all_last2 \
  --init-checkpoint outputs/speech_to_motion/base_all_frozen/checkpoints/last.pt \
  --batch-size 8 --accumulation-steps 4 --unfreeze-last-n 2
```

Defaults: BF16 on CUDA, decoder batch 32, AdamW, learning rate 1e-4, weight decay
0.01, dropout 0.1, gradient clipping 1.0. Loss is normalized-angle L1 plus 0.1
velocity-displacement L1. The epoch limits are 100 frozen and 30 fine-tuning.
**Training-only mode does not run validation or early stopping**, regardless of
`training.patience`. It trains to the configured epoch limit unless interrupted.

Fine-tuning bypasses cached features, with encoder LR 2e-6 and decoder LR 5e-5.
The convolutional frontend and earlier layers stay frozen. Non-reentrant gradient
checkpointing is enabled. Microbatch 8 × accumulation 4 gives an effective batch
of 32. Use `--unfreeze-last-n 4` in a separate run for a larger adaptation trial.
Use `--online` to train a frozen encoder without caching.

The A100 80 GB is the preferred lab GPU; profile before increasing batches. CPU
uses FP32. Full encoder/decoder weights, architecture, normalization, dataset ID,
optimizer/scaler and Torch/DataLoader RNG states are saved in checkpoints.

`last.pt` is the latest epoch; `best.pt` means **lowest training loss** in this
mode, not validated performance. Fine-tuning starts from `last.pt` by default.
Resume an interrupted run with `--resume .../checkpoints/last.pt`, the same run
name and stage, and the same training hyperparameters. A changed dataset identity
requires a new run. Existing standardized and split-run checkpoints are preserved.

## TensorBoard and outputs

```text
dataset/speech_to_motion_all/
    summary.json
    normalization.json
    splits/train.jsonl
    splits/val.jsonl                 # Empty
    splits/test.jsonl                # Empty
    hubert_features/...
outputs/speech_to_motion/<run_name>/
    config.yaml
    normalization.json
    dataset_summary.json
    metrics.jsonl
    tensorboard/{epochs,batches}/
    checkpoints/{best,last}.pt
```

New training-only runs log training loss, best training loss, learning rates,
degree MAE/RMSE, per-joint RMSE and percent-of-range errors, joint-limit violation
rate, and tolerance accuracy within 1/2/5/10 degrees. No validation/test metrics
are produced. Images include the joint RMSE table and tolerance accuracy matrix.
Per-joint `% of range = 100 * RMSE / (max - min)` is an error measure: lower is
better. Select the `base_all_*` run to see current training; historical runs keep
their earlier curves. See `runpod/README_speech_to_motion_lab.md` for SSH forwarding.

## Inference and Unity

```bash
python -m speech_to_motion.inference \
  --checkpoint outputs/speech_to_motion/base_all_last2/checkpoints/last.pt \
  --audio path/to/speech.wav --output outputs/speech_to_motion/demo_joints.csv
```

Inference blends overlapping windows, converts angles back to degrees using the
checkpoint's own normalization, and writes the existing timestamped 14-joint CSV
schema. `--clamp` applies Unity's physical limits. Existing output needs
`--overwrite`. Preview with Unity's **CommU → Preview Video Joint CSV…**. Unity
currently plays motion only; synchronized audio playback is a next integration.

## Optional legacy split experiments

`configs/split_minmax_legacy.yaml` retains source-grouped 80/10/10 experiments;
`configs/standardized_legacy.yaml` retains the original standardized version.
Their prepared directories and checkpoints remain separate. For those configs,
validation-based early stopping and held-out `speech_to_motion.evaluate` are
still supported. Evaluation rejects validation/test requests for all-data runs.

## Tests

`python -m pytest speech_to_motion/tests -q` uses synthetic audio/motion and a
tiny HuBERT, with no weight download. Tests cover normalization, degree metrics,
masking, source splits, all-data migration and cache reuse, both training stages,
resume, absence of validation calls/events in training-only mode, and CSV export.
