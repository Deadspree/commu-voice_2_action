# AGENTS.md — CommU Voice 2 Action

Working notes for the CommU pose-estimation project. Entries are tagged by date.

---

## 2026-08-21

### Joint-angle normalization (changed)
- Switched from a single global z-score (`angle_mean`/`angle_std`) to **per-joint min-max scaling to [-1, 1]** using physical joint limits.
- Limits live in `DataLoaderConfig.joint_limits` (tuple of `(min, max)` per joint), manually mirrored from `CS_JointLimitDefinition.cs`.
- **IMPORTANT:** Python limits are HARDCODED, not read from the C# file at runtime. Must keep in sync manually. ArmPitch max is **0** (not 5).
- Normalize: `2*(angle-min)/(max-min)-1`; Denormalize: `(norm+1)*(max-min)/2+min`.
- `CommuDataset._normalize_joints()` (instance) + `CommuDataset.denormalize_joints()` (static) in `commu_dataloader.py`.
- `evaluate.py` and `train.py` validation both denormalize to degrees for RMSE reporting (loss stays on normalized values).

### Augmentation (changed)
- Training transform now uses **RandomResizedCrop** (configurable via `train_use_rrc`, default **False** → plain Resize).
- `train_rrc_scale` loosened to `(0.7, 1.0)`; color jitter reduced to 0.15; blur sigma `(0.1, 1.5)`.
- `RandomHorizontalFlip` default OFF (`train_hflip_prob=0.0`) because robot arm joints are asymmetric.

### Regularization / model (changed)
- `ModelConfig.dropout` raised 0.1 → **0.3**.
- Added `ModelConfig.freeze_last_n` + `DINORegressor.freeze_backbone(freeze_last_n)` for **partial backbone freezing** (keep last N blocks + final norm trainable). Verified: `freeze_last_n=4` → 58/175 backbone params trainable.
- Added `--freeze-last-n` CLI arg to `train.py` (wired into model + config snapshot).

### Evaluation (changed)
- `evaluate.py` per-joint output now shows **Range**, **RMSE (deg)**, and **% of range** (`RMSE / (Max-Min) * 100`) so joints with different ranges are comparable.

### Key findings / diagnostics
- **Overfitting diagnosis:** train_loss ≈ 0.20 > val_loss ≈ 0.146 → model was *underfitting* the augmented data, not overfitting. Model had converged (val flat after epoch ~131).
- **Root cause discovered:** the **pretrained DINOv3 backbone weights were NOT loaded properly** in earlier runs — the model was training from random weights. This is the real bottleneck, not augmentation.
- **Action:** retrain from the properly-loaded pretrained backbone (not from current checkpoints built on random weights).

### Training setup / commands
- Conda env: `commu-voice-2-action` (must activate; base env has broken torchvision/torch mismatch).
- Two-step training: step 1 frozen backbone → step 2 unfreeze (partial via `--freeze-last-n`, or full).
- `--resume` = continue same run (restores optimizer/scheduler/epoch). `--load-weights` = load weights only, fresh optimizer (use when changing freeze config or architecture).
- Changing `hidden_dim` or `image_size` requires retraining (head shape / positional embeddings don't transfer).

### RunPod / SSH
- Active instance: `root@194.68.245.56 -p 22107 -i ~/.ssh/minhle`, remote dir `/workspace/commu-voice_2_action`.
- `runpod_dinov3.sh` (push) and `runpod_pull.sh` (pull) exist but `runpod_pull.sh` has stale connection settings (213.173.102.146:39323) — update if needed.
- Weights copied to pod: `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
- Pulled checkpoints: `step2_unfrozen_best.pt`, `step2_full_unfrozen_cont3_best.pt`.

### Next steps / open items
- **Verify pretrained weights load correctly** (check missing/unexpected keys) before next training.
- Retrain from pretrained backbone with RRC disabled (full images).
- Consider `hidden_dim=512` and `image_size=(384,384)` (source is 1920×1920) — both require retraining; may need smaller batch (e.g. 16).
- Add more data (2–3× current ~1,999 samples), prioritizing arm-pitch and mouth poses across full ranges.