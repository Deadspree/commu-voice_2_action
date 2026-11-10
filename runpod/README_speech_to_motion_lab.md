# Lab training on kaiso

Project: `/data1/minhle/Works/commu-voice_2_action`.
Connect from the Mac with `ssh -J lab-jump minhle@10.0.1.203`.

All project environments, caches, datasets and outputs live on `/data1` because
`/home` is full. No passwords or API credentials are stored with the project.
The project `.venv` reuses Torch 2.9.1+cu126 from
`/data1/minhle/miniforge3/envs/comfyvenv`; additional libraries are installed only
in `.venv`, and FFmpeg/ffprobe live in `.tools`. The original environment is not
modified. Keep it available because this venv depends on its Torch installation.
Do not install the root requirements' CUDA 12.8 wheels over this lab setup.

The launcher handles the Python environment, PATH, Hugging Face cache and
`CUDA_VISIBLE_DEVICES=3` with PCI bus ordering. Python calls the fourth physical
GPU `cuda:0`. The launcher does not reserve this shared GPU; check availability
with `nvidia-smi -i 3` before starting.

## Train all videos

The default config uses all 593 clips for training. Physical joint-limit
normalization is unchanged. There are no validation/test metrics or early
stopping. Epoch limits remain 100 frozen and 30 fine-tuning; patience is ignored
in this mode. Historical split experiments remain available separately.

```bash
cd /data1/minhle/Works/commu-voice_2_action
bash runpod/lab_speech_to_motion.sh check

# Only needed when the prepared dataset/window settings change:
bash runpod/lab_speech_to_motion.sh features

# Frozen decoder training:
bash runpod/lab_speech_to_motion.sh frozen

# After frozen training, fine-tune the last two HuBERT layers:
bash runpod/lab_speech_to_motion.sh finetune
```

These use new run names `base_all_frozen` and `base_all_last2`. Cached features
are reused; extra training-stride windows for former held-out clips are filled
once during setup. No source audio or CSV is duplicated. Migration preserves
`dataset/speech_to_motion_minmax/normalization.json` exactly in the new
`dataset/speech_to_motion_all` directory.

Use tmux for long jobs:

```bash
tmux new -s commu-training
cd /data1/minhle/Works/commu-voice_2_action
bash runpod/lab_speech_to_motion.sh frozen
# Detach: Ctrl+B, D. Reattach: tmux attach -t commu-training
```

To retrain again, override `--run-name` with a fresh name. When fine-tuning a
custom frozen run, also override `--init-checkpoint` to its `last.pt`.
`last.pt` is the latest training state; `best.pt` selects lowest training loss,
not held-out accuracy. The finetune launcher defaults to the frozen `last.pt`.

Resume frozen training:

```bash
bash runpod/lab_speech_to_motion.sh frozen \
  --resume outputs/speech_to_motion/base_all_frozen/checkpoints/last.pt
```

Resume fine-tuning using the module stage, without an initialization argument:

```bash
bash runpod/lab_speech_to_motion.sh module speech_to_motion.train \
  --stage finetune --run-name base_all_last2 \
  --resume outputs/speech_to_motion/base_all_last2/checkpoints/last.pt \
  --batch-size 8 --accumulation-steps 4 --unfreeze-last-n 2
```

## TensorBoard on the Mac

The running server is in tmux session `commu-tensorboard`. To start it again:

```bash
cd /data1/minhle/Works/commu-voice_2_action
tmux new -s commu-tensorboard
bash runpod/lab_speech_to_motion.sh tensorboard
```

On the Mac, from the local repository:

```bash
bash runpod/lab_tensorboard_tunnel.sh
```

Or from any Mac terminal:

```bash
ssh -N -T -J lab-jump -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:6006:127.0.0.1:6006 minhle@10.0.1.203
```

Keep the tunnel open and view http://localhost:6006. If occupied, forward Mac
port 16006 to the lab's port 6006 instead. The server listens only on lab
loopback, without a publicly exposed port.

Select `base_all_*` runs. Scalars show training losses, learning rates, degree
MAE/RMSE, per-joint RMSE and `% of range`, and tolerance accuracy. Images show
`rmse_by_joint/train` and `accuracy_matrix/train`. There are no validation/test
metrics in these runs. Historical split-run curves are retained in older runs.
