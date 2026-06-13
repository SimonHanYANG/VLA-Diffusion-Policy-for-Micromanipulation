# VLA: Vision-Language-Action Diffusion Policy for Micro-Object Navigation

A complete framework for **Vision-Language guided fine-grained micro-object positioning** using conditional diffusion policies. The system learns to navigate microscopic targets (embryos, oocytes, sperm heads, sperm tails) to a laser focal point in simulated microscope environments, with a sim-to-real transfer pipeline for deployment on real Nikon Ti2E microscopes.

## Overview

This project implements a **Diffusion Policy** based VLA (Vision-Language-Action) model that:

1. Takes microscope images + language instructions as input
2. Predicts smooth, multi-step action sequences (stage displacement in pixel space)
3. Uses closed-loop control to iteratively navigate targets to a laser dot
4. Transfers from simulation to real hardware via domain randomization + few-shot fine-tuning

### Key Design Decisions

- **Real cell images**: Uses pre-segmented real microscopy images (`data/pre-individual-obj/individual_obj/`) instead of procedural geometry for targets
- **Clean rendering**: No noise, no optical effects, no domain randomization — pure grayscale simulation matching real camera output
- **Resolution**: 640×640 pixels (matching real camera 1600×1200 center-crop deployment)
- **Scale factor**: All pixel parameters auto-scale based on resolution (reference: 224×224)

## Project Structure

```
VLA/
├── configs/                          # YAML configuration files
│   ├── model/
│   │   └── diffusion_policy.yaml     # Model hyperparameters
│   ├── simulator/
│   │   ├── clean_640.yaml            # Clean 640×640 simulator config (no noise/optics)
│   │   ├── embryo.yaml               # Embryo task (RealImageTargetGenerator)
│   │   ├── oocyte.yaml               # Oocyte task (RealImageTargetGenerator)
│   │   ├── real_sperm_head.yaml      # Sperm head task (RealImageTargetGenerator)
│   │   ├── whole_sperm.yaml          # Whole sperm task (SpermTailFromImageGenerator)
│   │   └── microsphere.yaml          # Microsphere task (procedural, no real images)
│   └── training/
│       └── default.yaml              # Training hyperparameters
├── data/
│   ├── pre-individual-obj/
│   │   └── individual_obj/           # Real cell images (PNG with alpha)
│   │       ├── embryo/               # ~149 embryo images
│   │       ├── oocyte/               # ~149 oocyte images
│   │       ├── sperm_head/           # ~149 sperm head images
│   │       └── whole_sperm/          # ~4023 whole sperm images
│   ├── backgrounds/                  # Real microscopy background patches (640×640)
│   ├── sperm_tail_tips.json          # Pre-computed tail tip positions
│   ├── trajectories/                 # Generated expert demonstrations
│   ├── checkpoints/                  # Saved model weights (.pt)
│   ├── runs/                         # TensorBoard logs
│   └── text_embeddings.pt            # Cached CLIP embeddings
├── scripts/
│   ├── generate_data.py              # Expert trajectory generation (main entry point)
│   ├── extract_backgrounds.py        # Extract background patches from video frames
│   ├── preprocess_sperm_tail_tips.py # Pre-compute sperm tail tip positions
│   ├── visualize_trajectories.py     # Trajectory visualization
│   ├── train.py                      # Training entry point
│   ├── eval.py                       # Batch evaluation
│   └── run_inference_panel.py        # Launch GUI for interactive testing
├── src/
│   ├── vla/                          # Core VLA model
│   │   ├── diffusion_policy.py       # Full diffusion policy (training + inference)
│   │   ├── diffusion_unet.py         # 1D U-Net for noise prediction
│   │   ├── visual_encoder.py         # ResNet-18 visual backbone
│   │   ├── text_encoder.py           # CLIP text encoder + caching
│   │   ├── condition_embed.py        # Visual-text fusion MLP
│   │   ├── noise_scheduler.py        # DDPM/DDIM noise schedule
│   │   └── inference.py              # Closed-loop controller
│   ├── simulator/                    # Microscope simulator
│   │   ├── environment.py            # Gym-like microscope env
│   │   ├── renderer.py               # Image composition pipeline
│   │   ├── targets.py                # Target generators (real image + procedural)
│   │   ├── stage.py                  # Simulated Nikon Ti2E stage
│   │   ├── background.py             # Background generators
│   │   ├── noise.py                  # Sensor noise models
│   │   ├── optics.py                 # Optical effects (PSF, vignetting)
│   │   └── expert_generator.py       # PID-based expert demonstrations
│   ├── data/
│   │   ├── dataset.py                # Trajectory dataset with sliding window
│   │   └── augmentation.py           # Flip, brightness, noise augmentation
│   ├── training/
│   │   ├── trainer.py                # Training loop with TensorBoard
│   │   ├── evaluator.py              # Batch policy evaluation
│   │   └── metrics.py                # Success rate, distance, smoothness
│   ├── ui/
│   │   ├── inference_panel.py        # Tkinter GUI for interactive testing
│   │   └── simulator_viewer.py       # OpenCV-based simulator viewer
│   ├── interfaces/
│   │   └── stage_interface.py        # Abstract stage API (sim ↔ real)
│   └── utils/
│       ├── config.py                 # Dataclass configs + YAML loaders
│       ├── logger.py                 # Logging setup
│       └── perlin.py                 # Perlin noise generator
├── tests/                            # Unit tests (44 tests)
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

### Step 0: Prepare Background Images

Extract clean background patches from real microscopy video frames:

```bash
python scripts/extract_backgrounds.py \
    --source-dir /path/to/microscopy/frames \
    --output-dir data/backgrounds \
    --patch-size 640 \
    --num-frames 200 \
    --patches-per-frame 5
```

This creates 640×640 grayscale background patches used by the simulator.

### Step 1: Generate Expert Demonstrations

The simulator uses a PID controller to automatically generate expert trajectories using real cell images:

```bash
# Generate data for all tasks (default: embryo, oocyte, real_sperm_head, whole_sperm, microsphere)
# Default: 100 trajectories per task, clean 640x640 config, real microscopy backgrounds
python scripts/generate_data.py --num-trajectories 100

# Generate data for a specific task
python scripts/generate_data.py --task embryo --num-trajectories 50

# Generate with multiple workers
python scripts/generate_data.py --num-trajectories 500 --workers 4
```

Each trajectory is saved as a folder containing:
- `images/` — sequence of JPEG frames (640×640 grayscale)
- `actions.npy` — (T, 2) array of (dx, dy) pixel displacements
- `positions.npy` — (T+1, 2) target positions
- `meta.json` — trajectory metadata

Generated data is saved to `data/trajectories/<task_name>/traj_NNNNN/`.

### Step 2: Pre-compute Text Embeddings

CLIP embeddings are computed once and cached:

```bash
python scripts/train.py --precompute-text --tasks all
```

This creates `data/text_embeddings.pt` containing 512-dim CLIP embeddings for each task instruction:
- `"Navigate the embryo to the red laser dot."`
- `"Navigate the oocyte to the red laser dot."`
- `"Navigate the sperm head to the red laser dot."`
- `"Navigate the sperm tail tip to the red laser dot."`
- `"Navigate the microsphere to the red laser dot."`

### Step 3: Train the Diffusion Policy

```bash
# Train on all tasks
python scripts/train.py --model diffusion_policy --tasks all

# Train on specific tasks
python scripts/train.py --model diffusion_policy --tasks embryo,oocyte

# Resume from checkpoint
python scripts/train.py --model diffusion_policy --tasks all --resume data/checkpoints/epoch_0100.pt

# Use GPU
python scripts/train.py --model diffusion_policy --tasks all --device cuda

# Custom image size (must match simulator)
python scripts/train.py --model diffusion_policy --tasks all --img-size 640
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
python scripts/eval.py --checkpoint data/checkpoints/best.pt --task embryo --episodes 100

# Evaluate on all tasks
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100

# Save results to JSON
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100 --output results.json
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
- **Center**: Real-time microscope view with trajectory overlay
- **Right panel**: Live metrics (distance, steps, success/fail), episode summary
- **Keyboard shortcuts**: `Space/S` = step, `A` = auto-run, `R` = reset, `1-4` = switch task, `Q/Esc` = quit

### Step 6: Visualize Trajectories

**Interactive UI Viewer** (recommended):

```bash
python scripts/visualize_trajectory_ui.py
```

| 按键 | 功能 |
|------|------|
| `Space` / `→` | 下一帧 |
| `←` | 上一帧 |
| `N` / `P` | 下一条 / 上一条轨迹 |
| `A` | 自动播放 / 暂停 |
| `R` | 重置到第一帧 |
| `1`-`5` | 切换任务 (embryo/oocyte/sperm_head/whole_sperm/microsphere) |
| `Q` / `Esc` | 退出 |

**Command-line Visualization**:

```bash
# Animate from a task directory (generate mp4)
python scripts/visualize_trajectories.py --task-dir data/trajectories/embryo --animate --frame-size 640 --no-overlay --output output/traj_embryo.mp4

# Grid view of multiple trajectories (generate png)
python scripts/visualize_trajectories.py --task-dir data/trajectories/embryo --num-show 6 --output output/grid.png
```

## Simulator Design

### Rendering Pipeline

The simulator renders synthetic microscope images that match real camera output:

1. **Background**: Load real microscopy background patches (grayscale, 640×640)
2. **Target**: Paste real cell image with edge blending (soft alpha, brightness adaptation)
3. **Laser**: Draw bright gray dot at image center (grayscale, not red)
4. **Output**: Pure grayscale 3-channel image (all channels identical)

**Important**: The simulator uses **no noise, no optical effects, no domain randomization** by default (`clean_640.yaml`). This produces clean training data that matches the real camera's grayscale output.

### Target Generators

| Task | Generator | Source | Reference Point |
|------|-----------|--------|-----------------|
| embryo | `RealImageTargetGenerator` | `data/pre-individual-obj/individual_obj/embryo/` | Image center |
| oocyte | `RealImageTargetGenerator` | `data/pre-individual-obj/individual_obj/oocyte/` | Image center |
| real_sperm_head | `RealImageTargetGenerator` | `data/pre-individual-obj/individual_obj/sperm_head/` | Image center |
| whole_sperm | `SpermTailFromImageGenerator` | `data/pre-individual-obj/individual_obj/whole_sperm/` | Tail tip |
| microsphere | `MicrosphereGenerator` | Procedural (no real images) | Circle center |

**Real image generators** load pre-segmented PNG images with alpha channels (alpha = mask), auto-scale to fit the target image, and apply edge blur for smooth blending.

**SpermTailFromImageGenerator** additionally loads pre-computed tail tip positions from `data/sperm_tail_tips.json` (generated by `scripts/preprocess_sperm_tail_tips.py`).

### Scale Factor Mechanism

All pixel-based parameters auto-scale based on resolution:

```python
scale_factor = min(image_size) / reference_size  # reference = 224
```

This ensures consistent physical proportions at any resolution:
- Start distance: 50-200 px × scale_factor
- Success tolerance: 2.0 px × scale_factor
- Action clipping: 30.0 px/step × scale_factor
- Target sizes: scaled proportionally

### Stage Simulation

- Mirrors the real Nikon Ti2E stage API (`src/interfaces/stage_interface.py`)
- Tracks position in micrometers with configurable range
- Pixel-to-micrometer conversion with ±10% random calibration error per episode

### Expert Demonstrations

- PID controller navigates target reference point to laser dot
- Random action noise added to simulate human operator jitter
- Action magnitude clipped to 30 px/step (scaled by resolution)
- Each trajectory records: images (JPEG), actions (npy), positions (npy), metadata (json)

## Architecture

The model follows a **conditional diffusion policy** architecture:

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

### Components

1. **Visual Encoder** (`src/vla/visual_encoder.py`): ResNet-18, first conv modified for 6 channels (2 stacked RGB frames), output 256-dim
2. **Text Encoder** (`src/vla/text_encoder.py`): CLIP ViT-B/32, pre-computed 512-dim embeddings
3. **Condition Embedding** (`src/vla/condition_embed.py`): Concat(visual, text) → MLP → 256-dim
4. **Diffusion U-Net** (`src/vla/diffusion_unet.py`): 1D U-Net with FiLM conditioning, input (B,2,10), output (B,2,10)
5. **Noise Scheduler** (`src/vla/noise_scheduler.py`): 100 training steps (cosine), 10 DDIM inference steps

### Inference (Closed-Loop Control)

```
obs_history = [env.reset()]
while not done:
    obs_seq = pad_and_stack(obs_history[-2:])  # last 2 frames
    action_seq = model.predict_action(obs_seq, text_emb)  # DDIM: 10 steps
    action = action_seq[0]  # take FIRST action only
    obs = env.step(action)
    obs_history.append(obs)
```

The model predicts a 10-step action sequence but only executes the first step (receding horizon).

## Configuration

### Simulator Config (`configs/simulator/clean_640.yaml`)

```yaml
image_size: [640, 640]
reference_size: 224
um_per_pixel: 0.5
max_steps_per_episode: 200
success_tolerance_px: 2.0
start_distance_min: 50.0
start_distance_max: 200.0

noise: null              # No noise
domain_randomization: null  # No domain randomization
optics: null             # No optical effects

background_type: "image"
background_image_dir: "data/backgrounds"
```

### Model Config (`configs/model/diffusion_policy.yaml`)

```yaml
obs_horizon: 2
pred_horizon: 10
action_dim: 2
visual_backbone: "resnet18"
visual_output_dim: 256
text_dim: 512
condition_hidden_dims: [512, 256]
condition_output_dim: 256
unet_dims: [64, 128, 256]
num_diffusion_steps: 100
num_ddim_steps: 10
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

## Testing

```bash
# Run all tests (44 tests)
pytest tests/ -v

# Run simulator tests only
pytest tests/test_simulator.py -v

# Run realistic sim tests only
pytest tests/test_realistic_sim.py -v

# Run VLA model tests (requires PyTorch)
pytest tests/test_vla.py -v
```

## Sim-to-Real Transfer

The framework is designed for two-stage deployment on real microscopes:

1. **Stage 1 — Simulation training**: Train with clean simulation data (640×640 grayscale) on 100+ generated trajectories per task.

2. **Stage 2 — Real-world fine-tuning**: Collect 10-20 real demonstrations on the Nikon Ti2E microscope, then fine-tune the pretrained model with a low learning rate.

**Deployment**: Center-crop 640×640 from the real camera's 1600×1200 image (no resize needed, preserves original pixel detail at 0.5 μm/pixel).

The `StageInterface` abstraction (`src/interfaces/stage_interface.py`) allows swapping between `StageSimulator` and a real `NikonTi2EStage` implementation with a single line change.

## License

[Your license here]
