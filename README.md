# commu-voice_2_action

## Setup

### 1. Install Miniforge3 (conda)

If you don't already have Miniforge3 installed, download and install it.

**Ubuntu / Linux (x86_64):**

```bash
# Download the installer
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Run the installer (accept the license and default install location ~/miniforge3)
bash Miniforge3-Linux-x86_64.sh
```

**Ubuntu / Linux (ARM64 / aarch64):**

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
```

**macOS (Apple Silicon / arm64):**

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

**macOS (Intel / x86_64):**

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh
```

After installation, restart your terminal (or run `source ~/.bashrc` on Ubuntu/Linux, or `source ~/.zshrc` on macOS) so `conda` is available.

> **Note:** If you prefer not to auto-activate conda's base environment on every shell, run `conda config --set auto_activate_base false` after installing.

### 2. Create and activate the conda environment

```bash
conda create -n commu-voice-2-action python=3.11 -y
conda activate commu-voice-2-action
```

### 3. Install the dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 4. Clone the DINOv3 repository

The DINOv3 backbone code lives in `third_party/dinov3`:

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

> **Note on pretrained weights:** The DINOv3 pretrained weights are gated. To use a pretrained backbone, request access at [https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Once approved, Meta emails you the download URLs. You can then either:
> - download the weights locally and pass the path via `backbone_weights`, or
> - pass the emailed URL directly via `backbone_weights`.
>
> Without access, the model falls back to a randomly initialized backbone.

## Usage

Train the model:

```bash
python pose_estimation/train.py --output-mode point --epochs 50
python pose_estimation/train.py --output-mode distribution --loss nll --epochs 50
```

Evaluate a trained checkpoint:

```bash
python pose_estimation/evaluate.py --checkpoint pose_estimation/checkpoints/best.pt
```
