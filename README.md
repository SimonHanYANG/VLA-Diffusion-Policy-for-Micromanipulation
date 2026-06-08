# VLA: Vision-Language-Action Diffusion Policy for Micro-Object Navigation

A complete framework for **Vision-Language guided fine-grained micro-object positioning** using conditional diffusion policies. The system learns to navigate microscopic targets (microspheres, yeast cells, sperm heads, sperm tails) to a laser focal point in simulated microscope environments, with a sim-to-real transfer pipeline for deployment on real Nikon Ti2E microscopes.

## Overview

This project implements a **Diffusion Policy** based VLA (Vision-Language-Action) model that:

1. Takes microscope images + language instructions as input
2. Predicts smooth, multi-step action sequences (stage displacement in pixel space)
3. Uses closed-loop control to iteratively navigate targets to a laser dot
4. Transfers from simulation to real hardware via domain randomization + few-shot fine-tuning

The framework handles both **rigid objects** (microspheres) and **deformable biological specimens** (yeast cells, sperm) with a unified architecture.

## Project Structure

```
VLA/
├── configs/                        # YAML configuration files
│   ├── model/
│   │   └── diffusion_policy.yaml   # Model hyperparameters
│   ├── simulator/
│   │   ├── default.yaml            # Default simulator settings
│   │   ├── microsphere.yaml        # Microsphere task config
│   │   ├── yeast.yaml              # Yeast task config
│   │   ├── sperm_head.yaml         # Sperm head task config
│   │   └── sperm_tail.yaml         # Sperm tail task config
│   └── training/
│       └── default.yaml            # Training hyperparameters
├── data/
│   ├── checkpoints/                # Saved model weights (.pt)
│   ├── runs/                       # TensorBoard logs
│   ├── trajectories/               # Generated expert demonstrations
│   │   ├── microsphere/
│   │   ├── yeast/
│   │   ├── sperm_head/
│   │   └── sperm_tail/
│   └── text_embeddings.pt          # Cached CLIP embeddings
├── scripts/
│   ├── generate_data.py            # Expert trajectory generation
│   ├── train.py                    # Training entry point
│   ├── eval.py                     # Batch evaluation
│   └── run_inference_panel.py      # Launch GUI for interactive testing
├── src/
│   ├── vla/                        # Core VLA model
│   │   ├── diffusion_policy.py     # Full diffusion policy (training + inference)
│   │   ├── diffusion_unet.py       # 1D U-Net for noise prediction
│   │   ├── visual_encoder.py       # ResNet-18 visual backbone
│   │   ├── text_encoder.py         # CLIP text encoder + caching
│   │   ├── condition_embed.py      # Visual-text fusion MLP
│   │   ├── noise_scheduler.py      # DDPM/DDIM noise schedule
│   │   └── inference.py            # Closed-loop controller
│   ├── simulator/                  # Microscope simulator
│   │   ├── environment.py          # Gym-like microscope env
│   │   ├── renderer.py             # Image composition pipeline
│   │   ├── targets.py              # Parametric target generators
│   │   ├── stage.py                # Simulated Nikon Ti2E stage
│   │   ├── background.py           # Perlin noise / image backgrounds
│   │   ├── noise.py                # Sensor noise models
│   │   └── expert_generator.py     # PID-based expert demonstrations
│   ├── data/
│   │   ├── dataset.py              # Trajectory dataset with sliding window
│   │   └── augmentation.py         # Flip, brightness, noise augmentation
│   ├── training/
│   │   ├── trainer.py              # Training loop with TensorBoard
│   │   ├── evaluator.py            # Batch policy evaluation
│   │   └── metrics.py              # Success rate, distance, smoothness
│   ├── ui/
│   │   ├── inference_panel.py      # Tkinter GUI for interactive testing
│   │   └── simulator_viewer.py     # OpenCV-based simulator viewer
│   ├── interfaces/
│   │   └── stage_interface.py      # Abstract stage API (sim ↔ real)
│   └── utils/
│       ├── config.py               # Dataclass configs + YAML loaders
│       ├── logger.py               # Logging setup
│       └── perlin.py               # Perlin noise generator
├── tests/                          # Unit tests
├── doc/
│   └── paper-design.md             # Research design document
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd VLA

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Model training and inference |
| `transformers` | CLIP text encoder (pre-computation only) |
| `opencv-python` | Image rendering and processing |
| `numpy` | Numerical computation |
| `pyyaml` | Configuration loading |
| `tensorboard` | Training visualization |
| `tqdm` | Progress bars |
| `Pillow` | GUI image display |
| `pytest` | Unit testing |

## Quick Start

### Step 1: Generate Expert Demonstrations

The simulator uses a PID controller to automatically generate expert trajectories:

```bash
# Generate data for a single task
python scripts/generate_data.py --task microsphere --num-trajectories 5000

# Generate data for all tasks (microsphere, yeast, sperm_head, sperm_tail)
python scripts/generate_data.py --all-tasks --num-trajectories 5000

# Parallel generation with multiple workers
python scripts/generate_data.py --all-tasks --num-trajectories 5000 --workers 4
```

Each trajectory is saved as a folder containing:
- `images/` — sequence of JPEG frames (224×224)
- `actions.npy` — (T, 2) array of (dx, dy) pixel displacements
- `positions.npy` — (T+1, 2) target positions
- `meta.json` — trajectory metadata

### Step 2: Pre-compute Text Embeddings

CLIP embeddings are computed once and cached:

```bash
python scripts/train.py --precompute-text --tasks all
```

This creates `data/text_embeddings.pt` containing 512-dim CLIP embeddings for each task instruction:
- `"Navigate the microsphere to the red laser dot."`
- `"Navigate the yeast cell to the red laser dot."`
- `"Navigate the sperm head to the red laser dot."`
- `"Navigate the sperm tail tip to the red laser dot."`

### Step 3: Train the Diffusion Policy

```bash
# Train on all tasks
python scripts/train.py --model diffusion_policy --tasks all

# Train on specific tasks
python scripts/train.py --model diffusion_policy --tasks microsphere,yeast

# Resume from checkpoint
python scripts/train.py --model diffusion_policy --tasks all --resume data/checkpoints/epoch_0100.pt

# Use GPU
python scripts/train.py --model diffusion_policy --tasks all --device cuda

# Custom configs
python scripts/train.py --model diffusion_policy --tasks all \
    --model-config configs/model/diffusion_policy.yaml \
    --training-config configs/training/default.yaml
```

Training outputs:
- `data/checkpoints/best.pt` — best validation loss checkpoint
- `data/checkpoints/epoch_NNNN.pt` — periodic checkpoints
- `data/runs/<timestamp>/` — TensorBoard logs

Monitor training in real-time:
```bash
tensorboard --logdir data/runs
```

### Step 4: Evaluate the Trained Model

```bash
# Evaluate on a single task (100 episodes)
python scripts/eval.py --checkpoint data/checkpoints/best.pt --task microsphere --episodes 100

# Evaluate on all tasks
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100

# Save results to JSON
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100 --output results.json

# Use GPU for faster inference
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --device cuda
```

Output metrics:
- **Success Rate** — fraction of episodes where final distance < 2 px
- **Mean Final Distance** — average pixel distance to laser dot at episode end
- **Trajectory Smoothness** — mean angular change between consecutive actions (lower = smoother)

### Step 5: Interactive Testing with GUI

```bash
# Launch the inference panel
python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt

# With GPU
python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt --device cuda
```

The GUI provides:
- **Left panel**: Task selector, noise/domain-rand toggles, speed control, step/auto buttons
- **Center**: Real-time microscope view with trajectory overlay (green), target crosshair (green), laser crosshair (red), action arrow (yellow)
- **Right panel**: Live metrics (distance, steps, success/fail), episode summary (success rate, avg distance, avg smoothness), predicted action (dx, dy)
- **Keyboard shortcuts**: `Space/S` = step, `A` = auto-run, `R` = reset, `1-4` = switch task, `Q/Esc` = quit

## Method Details

### Architecture

The model follows a **conditional diffusion policy** architecture with four main components:

```
                    ┌─────────────────┐
                    │  Language Inst.  │
                    │  (CLIP encoded)  │
                    └────────┬────────┘
                             │ text_emb (512-dim)
┌──────────────┐             │
│ Observation   │             │
│ History       │             │
│ (2 frames)    │             │
└──────┬───────┘             │
       │                      │
       ▼                      │
┌──────────────┐             │
│ Visual Encoder│             │
│ (ResNet-18)   │             │
│ → 256-dim     │             │
└──────┬───────┘             │
       │ visual_feat          │
       ▼                      ▼
┌──────────────────────────────┐
│    Condition Embedding        │
│    Concat → MLP → 256-dim    │
└──────────────┬───────────────┘
               │ condition c
               ▼
┌──────────────────────────────┐
│    1D Diffusion U-Net         │
│                              │
│  x_T ~ N(0,I) ──→ denoise   │
│  t (timestep)                │
│  c (condition)               │
│                              │
│  FiLM conditioning at        │
│  each down/up block          │
│                              │
│  → predicted noise           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    DDIM Sampler               │
│    10 steps (train: 100)     │
│    → action sequence          │
│    (10 steps × 2D)           │
└──────────────────────────────┘
```

#### 1. Visual Encoder (`src/vla/visual_encoder.py`)

- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Input modification**: First conv layer modified to accept `obs_horizon × 3` channels (2 RGB frames stacked = 6 channels). Pretrained weights are replicated across new channels.
- **Output**: 256-dim feature vector via `Linear → ReLU → LayerNorm`
- **Observation horizon**: 2 frames (provides temporal context for motion estimation)

#### 2. Text Encoder (`src/vla/text_encoder.py`)

- **Model**: CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)
- **Usage**: CLIP is used only once before training to precompute 512-dim embeddings for each task's fixed language instruction. The full CLIP model is **not** loaded during training or inference.
- **Caching**: Embeddings are saved to `data/text_embeddings.pt` and loaded as a simple dictionary.

#### 3. Condition Embedding (`src/vla/condition_embed.py`)

Fuses visual and text features into a single conditioning vector:

```
concat(visual_256, text_512) → Linear(768→512) → ReLU → Dropout(0.1)
                             → Linear(512→256) → ReLU → Dropout(0.1)
                             → Linear(256→256) → LayerNorm
```

Output: 256-dim condition vector `c` that drives the diffusion U-Net via FiLM modulation.

#### 4. Diffusion U-Net (`src/vla/diffusion_unet.py`)

A **1D convolutional U-Net** that predicts noise from noisy action sequences:

- **Input**: Noisy actions `(B, 2, 10)` + timestep `t` + condition `c`
- **Output**: Predicted noise `(B, 2, 10)`
- **Architecture**:
  - **Timestep embedding**: Sinusoidal positional encoding → MLP → 256-dim
  - **Combined conditioning**: `[time_emb; cond]` → Linear → 256-dim FiLM vector
  - **Encoder**: `Conv1d(2→64) → DownBlock(64→128) → DownBlock(128→256)`
  - **Bottleneck**: Two Conv1d layers with FiLM conditioning
  - **Decoder**: `UpBlock(256+128→128) → UpBlock(128+64→64)` with skip connections
  - **Output**: `Conv1d(64→2)`, interpolated to match horizon
- **FiLM conditioning**: At each block, the condition vector produces scale and shift parameters: `output = x * (1 + scale) + shift`
- **DownBlock**: `Conv1d(stride=2) → GroupNorm → ReLU → Conv1d → GroupNorm → FiLM → ReLU + residual`
- **UpBlock**: `Upsample(2×) → cat(skip) → Conv1d → GroupNorm → ReLU → Conv1d → GroupNorm → FiLM → ReLU + residual`

#### 5. Noise Scheduler (`src/vla/noise_scheduler.py`)

- **Training schedule**: 100 diffusion steps with cosine beta schedule
- **Inference schedule**: DDIM with 10 steps (10× faster than DDPM)
- **Forward diffusion**: `x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε`
- **DDIM step**: Deterministic reverse process — predicts `x_0` from `x_t` and noise prediction, then computes `x_{t-1}`

### Training Procedure

**Loss**: Standard denoising score matching — MSE between predicted noise and actual noise:

```python
# Pseudocode
visual_feat = visual_encoder(obs_seq)           # (B, 256)
cond = condition_embed(visual_feat, text_emb)   # (B, 256)
noise = randn(B, 2, 10)                         # random noise
t = randint(0, 100)                             # random timestep
noisy_actions = scheduler.add_noise(actions, noise, t)
noise_pred = unet(noisy_actions, t, cond)
loss = MSE(noise_pred, noise)
```

**Optimizer**: AdamW (lr=1e-4, weight_decay=1e-6)

**Scheduler**: Cosine decay with linear warmup (10 epochs warmup, 500 total epochs)

**Data augmentation** (`src/data/augmentation.py`):
- Random horizontal flip (p=0.5) — flips both images and x-action
- Random brightness/contrast jitter
- Gaussian noise injection

**Training tricks**:
- Gradient accumulation support
- Early stopping on validation loss (patience=100)
- Periodic checkpointing every 50 epochs
- TensorBoard logging (loss, learning rate)

### Inference (Closed-Loop Control)

At test time, the model runs in a **closed-loop** fashion:

```
obs_history = [env.reset()]
while not done:
    obs_seq = pad_and_stack(obs_history[-2:])  # last 2 frames
    action_seq = model.predict_action(obs_seq, text_emb)  # DDIM: 10 steps
    action = action_seq[0]  # take FIRST action only
    obs = env.step(action)
    obs_history.append(obs)
```

**Key design**: The model predicts a 10-step action sequence but only executes the first step. The next observation triggers a new prediction. This **receding horizon** approach provides robustness to compounding errors.

### Simulator Design (`src/simulator/`)

The simulator renders synthetic microscope images with full control over visual appearance:

**Rendering pipeline** (`renderer.py`):
1. Generate Perlin noise background (or load real image crops)
2. Paste target object mask at specified position (alpha blending)
3. Draw red laser dot at image center (radius=3 px)
4. Apply domain randomization (brightness, contrast, Gaussian blur)
5. Apply sensor noise (Gaussian, salt-pepper, flicker, motion blur)

**Target generators** (`targets.py`):
| Target | Shape | Size | Reference Point |
|--------|-------|------|-----------------|
| Microsphere | Circle (solid or ring) | r=15-25 px | Center |
| Yeast | Ellipse + sinusoidal bumps | major=20-35 px | Center |
| Sperm head | Ellipse + tapered tip | major=18-28 px | Center |
| Sperm tail | Cubic Bezier curve + head | length=40-100 px | Tip endpoint |

**Stage simulation** (`stage.py`):
- Mirrors the real Nikon Ti2E stage API (abstract interface in `interfaces/stage_interface.py`)
- Tracks position in micrometers with configurable range
- Pixel-to-micrometer conversion with ±10% random calibration error per episode

**Expert demonstrations** (`expert_generator.py`):
- PID controller navigates target reference point to laser dot
- Random action noise added to simulate human operator jitter
- Action magnitude clipped to 30 px/step
- Each trajectory records: images (JPEG), actions (npy), positions (npy), metadata (json)

### Evaluation Metrics (`src/training/metrics.py`)

| Metric | Definition |
|--------|------------|
| **Success Rate** | Fraction of episodes with final distance < 2 px |
| **Mean Final Distance** | Average Euclidean distance (px) from target to laser at episode end |
| **Trajectory Smoothness** | Mean absolute angular change between consecutive action vectors (range [0, π], lower = smoother) |
| **Trajectory Length** | Number of steps until done |

## Configuration

### Model Config (`configs/model/diffusion_policy.yaml`)

```yaml
obs_horizon: 2              # Number of observation frames
pred_horizon: 10            # Action sequence length
action_dim: 2               # (dx, dy) displacement
visual_backbone: "resnet18"
visual_output_dim: 256
text_dim: 512               # CLIP embedding dimension
condition_hidden_dims: [512, 256]
condition_output_dim: 256
unet_dims: [64, 128, 256]   # U-Net channel progression
num_diffusion_steps: 100    # Training diffusion steps
num_ddim_steps: 10          # DDIM inference steps
beta_schedule: "cosine"
```

### Training Config (`configs/training/default.yaml`)

```yaml
batch_size: 64
learning_rate: 0.0001
weight_decay: 0.000001
num_epochs: 500
warmup_epochs: 10
grad_accum_steps: 1
val_split: 0.1
val_freq_epochs: 10
save_freq_epochs: 50
early_stopping_patience: 100
log_dir: "data/runs"
```

### Simulator Config (`configs/simulator/default.yaml`)

```yaml
image_size: [224, 224]
um_per_pixel: 0.5
um_per_pixel_noise: 0.1        # ±10% calibration error
max_steps_per_episode: 200
success_tolerance_px: 2.0
start_distance_min: 50.0       # Min start distance from laser
start_distance_max: 200.0      # Max start distance from laser

noise:
  gaussian_std: 5.0
  salt_pepper_prob: 0.02
  flicker_prob: 0.1
  motion_blur_prob: 0.05

domain_randomization:
  brightness_range: [0.5, 1.5]
  contrast_range: [0.5, 1.5]
  blur_sigma_max: 2.0
  blur_prob: 0.3
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_simulator.py
pytest tests/test_vla.py
```

## Sim-to-Real Transfer

The framework is designed for two-stage deployment on real microscopes:

1. **Stage 1 — Simulation training**: Train with full domain randomization (backgrounds, lighting, noise, calibration error) on 5000+ generated trajectories per task.

2. **Stage 2 — Real-world fine-tuning**: Collect 10-20 real demonstrations on the Nikon Ti2E microscope, then fine-tune the pretrained model with a low learning rate for 50-100 epochs. Optionally freeze the visual encoder and only fine-tune the U-Net action head.

The `StageInterface` abstraction (`src/interfaces/stage_interface.py`) allows swapping between `StageSimulator` and a real `NikonTi2EStage` implementation with a single line change in the environment constructor.

## License

[Your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_citation,
  title={Vision-Language-Action Diffusion Policy for Micro-Object Navigation},
  author={Simon Yang},
  year={2026}
}
```
