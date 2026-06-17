"""Tkinter-based inference panel for post-training VLA model testing.

Load a trained model checkpoint, generate random microscope environments,
run closed-loop inference, and visualize target movement and model predictions.

Usage:
  python scripts/run_inference_panel.py
  python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.simulator.environment import MicroscopeEnvironment
from src.simulator.targets import create_target_generator, get_target_generator_for_task
from src.utils.config import load_task_config
from src.training.metrics import EpisodeMetrics
from src.utils.config import (
    DiffusionPolicyConfig,
    SimulatorConfig,
    load_diffusion_policy_config,
    load_simulator_config,
)
from src.vla.diffusion_policy import DiffusionPolicy
from src.vla.text_encoder import CachedTextEmbeddings, TextEncoder

TASK_TYPES = ["embryo", "oocyte", "real_sperm_head", "whole_sperm", "microsphere"]
DEFAULT_MODEL_CONFIG = "configs/model/diffusion_policy.yaml"
DEFAULT_SIM_CONFIG = "configs/simulator/clean_640.yaml"
DEFAULT_TEXT_CACHE = "data/text_embeddings.pt"


class InferencePanel:
    """Desktop UI panel for model inference testing on the microscope simulator."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_config_path: str = DEFAULT_MODEL_CONFIG,
        sim_config_path: str = DEFAULT_SIM_CONFIG,
        text_cache_path: str = DEFAULT_TEXT_CACHE,
        device: str = "cpu",
    ):
        self.model_config_path = model_config_path
        self.sim_config_path = sim_config_path
        self.text_cache_path = text_cache_path
        self.device = device

        # Model / env state
        self.model: Optional[DiffusionPolicy] = None
        self.model_config: Optional[DiffusionPolicyConfig] = None
        self.text_cache: Optional[CachedTextEmbeddings] = None
        self.sim_config: SimulatorConfig = load_simulator_config(Path(sim_config_path))
        self.env: Optional[MicroscopeEnvironment] = None
        self.obs: Optional[np.ndarray] = None
        self.info: dict = {}

        # Inference state
        self.task_idx = 0
        self.current_seed = 42
        self.auto_mode = False
        self.step_delay_ms = 100
        self.traj_points: list[tuple] = []
        self.actions: list[np.ndarray] = []
        self.done = False
        self.step_count = 0

        # Episode accumulator
        self.episode_total = 0
        self.episode_successes = 0
        self.episode_distances: list[float] = []
        self.episode_smoothnesses: list[float] = []

        # FPS tracking
        self._last_frame_time = time.time()
        self._fps = 0.0

        # Build UI
        self.root = tk.Tk()
        self.root.title("VLA Inference Panel")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_menu()
        self._setup_ui()
        self._bind_keys()

        # Load model if provided
        if checkpoint_path:
            self._load_model(checkpoint_path)
            self._init_env()

        # If no model loaded, still init env for manual viewing
        if self.env is None:
            self._init_env()

        # Start update loop
        self._update_loop()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Model...", command=self._on_open_model, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Sim Config...", command=self._on_open_sim_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close, accelerator="Ctrl+Q")

        menubar.add_cascade(label="File", menu=file_menu)
        self.root.bind_all("<Control-o>", lambda e: self._on_open_model())
        self.root.bind_all("<Control-q>", lambda e: self._on_close())

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        # Main horizontal paned window
        self.pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.pw.pack(fill=tk.BOTH, expand=True)

        # --- Left: Control Panel ---
        self.left_frame = ttk.Frame(self.pw, width=220)
        self.pw.add(self.left_frame, weight=0)
        self._setup_control_panel()

        # --- Center: Image Display ---
        self.center_frame = ttk.Frame(self.pw)
        self.pw.add(self.center_frame, weight=1)
        self._setup_image_panel()

        # --- Right: Metrics Panel ---
        self.right_frame = ttk.Frame(self.pw, width=220)
        self.pw.add(self.right_frame, weight=0)
        self._setup_metrics_panel()

        # --- Bottom: Status Bar ---
        self.status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._setup_status_bar()

    def _setup_control_panel(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # Header
        hdr = ttk.Label(self.left_frame, text="Control", font=("TkDefaultFont", 11, "bold"))
        hdr.pack(fill=tk.X, **pad)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Task selector
        ttk.Label(self.left_frame, text="Task:").pack(anchor=tk.W, **pad)
        self.task_var = tk.StringVar(value=TASK_TYPES[0])
        self.task_combo = ttk.Combobox(
            self.left_frame, textvariable=self.task_var,
            values=TASK_TYPES, state="readonly",
        )
        self.task_combo.pack(fill=tk.X, **pad)
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_change)

        # Language instruction display
        ttk.Label(self.left_frame, text="Language Instruction:", font=("TkDefaultFont", 8, "bold")).pack(anchor=tk.W, **pad)
        self.instruction_label = ttk.Label(
            self.left_frame,
            text=self._get_instruction_text(TASK_TYPES[0]),
            foreground="blue",
            wraplength=200,
            justify=tk.LEFT,
        )
        self.instruction_label.pack(anchor=tk.W, padx=12, pady=(0, 4))

        # Text embedding status
        self.text_status_label = ttk.Label(self.left_frame, text="Text Emb: not loaded", foreground="gray")
        self.text_status_label.pack(anchor=tk.W, **pad)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Seed
        ttk.Label(self.left_frame, text="Seed:").pack(anchor=tk.W, **pad)
        self.seed_label = ttk.Label(self.left_frame, text=str(self.current_seed), foreground="gray")
        self.seed_label.pack(anchor=tk.W, **pad)

        # Noise toggle
        self.noise_var = tk.BooleanVar(value=self.sim_config.noise is not None)
        self.noise_cb = ttk.Checkbutton(
            self.left_frame, text="Sensor Noise", variable=self.noise_var,
            command=self._on_noise_toggle,
        )
        self.noise_cb.pack(anchor=tk.W, **pad)

        # Domain rand toggle
        self.dr_var = tk.BooleanVar(value=self.sim_config.domain_randomization is not None)
        self.dr_cb = ttk.Checkbutton(
            self.left_frame, text="Domain Randomization", variable=self.dr_var,
            command=self._on_dr_toggle,
        )
        self.dr_cb.pack(anchor=tk.W, **pad)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Action buttons
        self.reset_btn = ttk.Button(self.left_frame, text="Reset Env", command=self._on_reset)
        self.reset_btn.pack(fill=tk.X, **pad)

        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(fill=tk.X, **pad)
        self.step_btn = ttk.Button(btn_frame, text="Step Once", command=self._on_step)
        self.step_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.auto_btn = ttk.Button(btn_frame, text="Auto Run", command=self._on_auto_toggle)
        self.auto_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Speed slider
        ttk.Label(self.left_frame, text="Speed (ms delay):").pack(anchor=tk.W, **pad)
        self.speed_var = tk.IntVar(value=self.step_delay_ms)
        self.speed_scale = ttk.Scale(
            self.left_frame, from_=1, to=500, variable=self.speed_var,
            orient=tk.HORIZONTAL, command=self._on_speed_change,
        )
        self.speed_scale.pack(fill=tk.X, **pad)

        # Speed indicator
        self.speed_label = ttk.Label(self.left_frame, text=f"{self.step_delay_ms} ms", foreground="gray")
        self.speed_label.pack(anchor=tk.W, **pad)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Model status
        self.model_status_label = ttk.Label(self.left_frame, text="Model: none", foreground="red")
        self.model_status_label.pack(anchor=tk.W, **pad)

    def _setup_image_panel(self) -> None:
        self.image_label = ttk.Label(self.center_frame, background="#1a1a2e")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _setup_metrics_panel(self) -> None:
        pad = {"padx": 8, "pady": 4}

        hdr = ttk.Label(self.right_frame, text="Metrics", font=("TkDefaultFont", 11, "bold"))
        hdr.pack(fill=tk.X, **pad)

        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Current episode
        ttk.Label(self.right_frame, text="Current Episode", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, **pad)

        frame1 = ttk.Frame(self.right_frame)
        frame1.pack(fill=tk.X, **pad)
        ttk.Label(frame1, text="Distance:").pack(side=tk.LEFT)
        self.dist_label = ttk.Label(frame1, text="-- px", foreground="gray")
        self.dist_label.pack(side=tk.RIGHT)

        frame2 = ttk.Frame(self.right_frame)
        frame2.pack(fill=tk.X, **pad)
        ttk.Label(frame2, text="Steps:").pack(side=tk.LEFT)
        self.steps_label = ttk.Label(frame2, text="0", foreground="gray")
        self.steps_label.pack(side=tk.RIGHT)

        frame3 = ttk.Frame(self.right_frame)
        frame3.pack(fill=tk.X, **pad)
        ttk.Label(frame3, text="Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(frame3, text="--", foreground="gray")
        self.status_label.pack(side=tk.RIGHT)

        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Accumulated stats
        ttk.Label(self.right_frame, text="Episode Summary", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, **pad)

        frame4 = ttk.Frame(self.right_frame)
        frame4.pack(fill=tk.X, **pad)
        ttk.Label(frame4, text="Total Episodes:").pack(side=tk.LEFT)
        self.ep_count_label = ttk.Label(frame4, text="0", foreground="gray")
        self.ep_count_label.pack(side=tk.RIGHT)

        frame5 = ttk.Frame(self.right_frame)
        frame5.pack(fill=tk.X, **pad)
        ttk.Label(frame5, text="Success Rate:").pack(side=tk.LEFT)
        self.sr_label = ttk.Label(frame5, text="--", foreground="gray")
        self.sr_label.pack(side=tk.RIGHT)

        frame6 = ttk.Frame(self.right_frame)
        frame6.pack(fill=tk.X, **pad)
        ttk.Label(frame6, text="Avg Final Dist:").pack(side=tk.LEFT)
        self.avg_dist_label = ttk.Label(frame6, text="-- px", foreground="gray")
        self.avg_dist_label.pack(side=tk.RIGHT)

        frame7 = ttk.Frame(self.right_frame)
        frame7.pack(fill=tk.X, **pad)
        ttk.Label(frame7, text="Avg Smoothness:").pack(side=tk.LEFT)
        self.avg_smooth_label = ttk.Label(frame7, text="--", foreground="gray")
        self.avg_smooth_label.pack(side=tk.RIGHT)

        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Language condition info
        ttk.Label(self.right_frame, text="Language Condition", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, **pad)

        self.cond_instruction_label = ttk.Label(
            self.right_frame,
            text="--",
            foreground="gray",
            wraplength=200,
            justify=tk.LEFT,
        )
        self.cond_instruction_label.pack(anchor=tk.W, padx=8, pady=(0, 2))

        frame_cond = ttk.Frame(self.right_frame)
        frame_cond.pack(fill=tk.X, **pad)
        ttk.Label(frame_cond, text="Emb Source:").pack(side=tk.LEFT)
        self.cond_source_label = ttk.Label(frame_cond, text="--", foreground="gray")
        self.cond_source_label.pack(side=tk.RIGHT)

        frame_dim = ttk.Frame(self.right_frame)
        frame_dim.pack(fill=tk.X, **pad)
        ttk.Label(frame_dim, text="Emb Dim:").pack(side=tk.LEFT)
        self.cond_dim_label = ttk.Label(frame_dim, text="--", foreground="gray")
        self.cond_dim_label.pack(side=tk.RIGHT)

        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)

        # Predicted action
        ttk.Label(self.right_frame, text="Predicted Action", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, **pad)
        self.action_label = ttk.Label(self.right_frame, text="dx: --  dy: --", foreground="gray")
        self.action_label.pack(anchor=tk.W, **pad)

    def _setup_status_bar(self) -> None:
        self.status_model = ttk.Label(self.status_frame, text="Model: --", relief=tk.SUNKEN)
        self.status_model.pack(side=tk.LEFT, padx=2)

        self.status_device = ttk.Label(self.status_frame, text=f"Device: {self.device}", relief=tk.SUNKEN)
        self.status_device.pack(side=tk.LEFT, padx=2)

        self.status_fps = ttk.Label(self.status_frame, text="FPS: --", relief=tk.SUNKEN)
        self.status_fps.pack(side=tk.RIGHT, padx=2)

    # ------------------------------------------------------------------
    # Language instruction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_instruction_text(task_name: str) -> str:
        """Return the language instruction for a given task name."""
        return TextEncoder.TASK_INSTRUCTIONS.get(task_name, f"Navigate the {task_name} to the red laser dot.")

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-space>", lambda e: self._on_step())
        self.root.bind("<KeyPress-a>", lambda e: self._on_auto_toggle())
        self.root.bind("<KeyPress-r>", lambda e: self._on_reset())
        self.root.bind("<KeyPress-s>", lambda e: self._on_step())
        self.root.bind("<KeyPress-q>", lambda e: self._on_close())
        self.root.bind("<KeyPress-Escape>", lambda e: self._on_close())
        self.root.bind("1", lambda e: self._set_task(0))
        self.root.bind("2", lambda e: self._set_task(1))
        self.root.bind("3", lambda e: self._set_task(2))
        self.root.bind("4", lambda e: self._set_task(3))
        self.root.bind("5", lambda e: self._set_task(4))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str) -> None:
        """Load a DiffusionPolicy from a checkpoint file."""
        try:
            self.model_config = load_diffusion_policy_config(Path(self.model_config_path))
            cfg = self.model_config

            self.model = DiffusionPolicy(
                obs_horizon=cfg.obs_horizon,
                pred_horizon=cfg.pred_horizon,
                action_dim=cfg.action_dim,
                visual_backbone=cfg.visual_backbone,
                visual_output_dim=cfg.visual_output_dim,
                text_dim=cfg.text_dim,
                condition_hidden_dims=cfg.condition_hidden_dims,
                condition_output_dim=cfg.condition_output_dim,
                unet_dims=cfg.unet_dims,
                num_diffusion_steps=cfg.num_diffusion_steps,
                num_ddim_steps=cfg.num_ddim_steps,
                beta_schedule=cfg.beta_schedule,
            )

            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Load text embeddings
            if Path(self.text_cache_path).exists():
                self.text_cache = CachedTextEmbeddings(self.text_cache_path)
                loaded_tasks = [t for t in TASK_TYPES if t in self.text_cache]
                self.text_status_label.config(
                    text=f"Text Emb: {len(loaded_tasks)}/{len(TASK_TYPES)} tasks loaded",
                    foreground="green",
                )
                print(f"Text embeddings loaded: {loaded_tasks}")
            else:
                self.text_cache = None
                self.text_status_label.config(
                    text=f"Text Emb: cache not found",
                    foreground="red",
                )
                print(f"Warning: text cache not found at {self.text_cache_path}")

            short_path = Path(checkpoint_path).name
            self.model_status_label.config(text=f"Model: {short_path}", foreground="green")
            self.status_model.config(text=f"Model: {short_path}")
            self._update_instruction_display()
            print(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model:\n{e}")
            self.model = None
            self.model_status_label.config(text="Model: error", foreground="red")

    def _on_open_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Model Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
            initialdir="data/checkpoints",
        )
        if path:
            self._load_model(path)
            self._init_env()
            self._on_reset()

    def _on_open_sim_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Simulator Config",
            filetypes=[("YAML", "*.yaml *.yml"), ("All Files", "*.*")],
            initialdir="configs/simulator",
        )
        if path:
            try:
                self.sim_config = load_simulator_config(Path(path))
                self.sim_config_path = path
                self._init_env()
                self._on_reset()
                print(f"Loaded sim config: {path}")
            except Exception as e:
                messagebox.showerror("Config Error", str(e))

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _init_env(self) -> None:
        """Create the microscope environment and inference controller."""
        task_name = TASK_TYPES[self.task_idx]
        # Try loading from YAML config first, fallback to built-in generator
        cfg_path = Path(f"configs/simulator/{task_name}.yaml")
        if cfg_path.exists():
            task_config = load_task_config(cfg_path)
            target_gen = create_target_generator(task_config.target_generator, task_config.target_params)
        else:
            target_gen = get_target_generator_for_task(task_name)
        self.env = MicroscopeEnvironment(config=self.sim_config, target_generator=target_gen)

        # If model loaded with text cache, we could use ClosedLoopController,
        # but for flexibility we manage inference manually in _model_step()
        self.obs, self.info = self.env.reset(seed=self.current_seed)
        self.traj_points = []
        self.actions.clear()
        self.done = False
        self.step_count = 0
        self.obs_history: list[np.ndarray] = [self.obs]

    def _on_reset(self) -> None:
        """Reset the environment with a new random seed."""
        self.current_seed = int(np.random.randint(0, 2**31 - 1))
        self.seed_label.config(text=str(self.current_seed))
        self._reset_env()

    def _reset_env(self) -> None:
        if self.env is None:
            return
        self.obs, self.info = self.env.reset(seed=self.current_seed)
        self.traj_points = []
        self.actions.clear()
        self.done = False
        self.step_count = 0
        self.obs_history = [self.obs]
        # Update display immediately
        self._update_image_display()
        self._update_metrics_display()

    def _update_instruction_display(self) -> None:
        """Update all UI elements related to language instruction and text embeddings."""
        task_name = TASK_TYPES[self.task_idx]
        instruction = self._get_instruction_text(task_name)

        # Left panel: instruction text
        self.instruction_label.config(text=instruction)

        # Right panel: condition info
        self.cond_instruction_label.config(text=f'"{instruction}"')

        # Check if this task has a cached embedding
        if self.text_cache and task_name in self.text_cache:
            emb = self.text_cache.get(task_name)
            self.cond_source_label.config(text="CLIP cache", foreground="green")
            self.cond_dim_label.config(text=str(emb.shape[0]), foreground="green")
        elif self.text_cache is not None:
            self.cond_source_label.config(text="zeros (no emb)", foreground="orange")
            self.cond_dim_label.config(text="512 (fallback)", foreground="orange")
        else:
            self.cond_source_label.config(text="no cache loaded", foreground="red")
            self.cond_dim_label.config(text="--", foreground="red")

    def _on_task_change(self, event=None) -> None:
        self.task_idx = TASK_TYPES.index(self.task_var.get())
        self._update_instruction_display()
        self._init_env()
        self._on_reset()

    def _on_noise_toggle(self) -> None:
        from src.utils.config import NoiseConfig
        if self.noise_var.get():
            self.sim_config.noise = NoiseConfig()
        else:
            self.sim_config.noise = None
        self._init_env()

    def _on_dr_toggle(self) -> None:
        from src.utils.config import DomainRandConfig
        if self.dr_var.get():
            self.sim_config.domain_randomization = DomainRandConfig()
        else:
            self.sim_config.domain_randomization = None
        self._init_env()

    def _on_speed_change(self, value) -> None:
        self.step_delay_ms = int(float(value))
        self.speed_label.config(text=f"{self.step_delay_ms} ms")

    def _set_task(self, idx: int) -> None:
        self.task_idx = idx
        self.task_var.set(TASK_TYPES[idx])
        self._update_instruction_display()
        self._init_env()
        self._on_reset()

    # ------------------------------------------------------------------
    # Step & Auto modes
    # ------------------------------------------------------------------

    def _on_step(self) -> None:
        """Manual single inference step."""
        if self.done or self.env is None:
            return
        self._step()

    def _step(self) -> None:
        """Run one step: inference + env step."""
        if self.env is None:
            return

        action = self._model_step()

        self.obs, reward, done, self.info = self.env.step(action)
        self.obs_history.append(self.obs)
        # Keep history bounded
        max_history = max(self.model_config.obs_horizon if self.model_config else 2, 10)
        if len(self.obs_history) > max_history:
            self.obs_history = self.obs_history[-max_history:]
        self.actions.append(action)
        self.step_count += 1

        tp = self.info.get("target_position", np.zeros(2))
        self.traj_points.append((int(tp[0]), int(tp[1])))

        if done:
            self.done = True
            self.auto_mode = False
            self.auto_btn.config(text="Auto Run")
            self._record_episode()

        self._update_image_display()
        self._update_metrics_display()

    def _model_step(self) -> Optional[np.ndarray]:
        """Run model inference. Returns predicted action (dx, dy) or None."""
        if self.model is None:
            # Fallback: use a zero action (no movement) for manual viewing
            return np.array([0.0, 0.0], dtype=np.float64)

        obs_horizon = self.model_config.obs_horizon if self.model_config else 2

        # Build observation sequence (pad if needed)
        if len(self.obs_history) < obs_horizon:
            pad = self.obs_history[0]
            obs_seq_imgs = [pad] * (obs_horizon - len(self.obs_history)) + list(self.obs_history)
        else:
            obs_seq_imgs = list(self.obs_history[-obs_horizon:])

        # Stack, transpose to (C, H, W), add batch dim, normalize
        obs_seq = torch.from_numpy(
            np.stack([img.transpose(2, 0, 1) for img in obs_seq_imgs], axis=0)
        ).float().unsqueeze(0).to(self.device) / 255.0

        # Text embedding
        task_name = TASK_TYPES[self.task_idx]
        if self.text_cache and task_name in self.text_cache:
            text_emb = self.text_cache.get(task_name).unsqueeze(0).to(self.device)
            # Update condition panel — using cached CLIP embedding
            self.cond_source_label.config(text="CLIP cache ✓", foreground="green")
        else:
            text_emb = torch.zeros(1, 512, device=self.device)
            # Update condition panel — fallback to zeros
            self.cond_source_label.config(text="zeros (fallback) ✗", foreground="red")

        with torch.no_grad():
            action_seq = self.model.predict_action(obs_seq, text_emb)
        return action_seq[0, 0].cpu().numpy()

    def _on_auto_toggle(self) -> None:
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            if self.done:
                self._on_reset()
            self.auto_btn.config(text="Stop")
        else:
            self.auto_btn.config(text="Auto Run")

    def _record_episode(self) -> None:
        """Record completed episode in accumulator."""
        self.episode_total += 1
        final_dist = float(self.info.get("distance", float("inf")))
        is_success = final_dist < self.sim_config.success_tolerance_px
        if is_success:
            self.episode_successes += 1
        self.episode_distances.append(final_dist)

        if self.actions:
            actions_arr = np.array(self.actions)
            smoothness = EpisodeMetrics.compute_smoothness(actions_arr)
        else:
            smoothness = 0.0
        self.episode_smoothnesses.append(smoothness)

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def _update_loop(self) -> None:
        """Main update loop, called every ~50ms via tkinter after()."""
        if self.auto_mode and not self.done and self.env is not None:
            self._step()

        # FPS
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self.status_fps.config(text=f"FPS: {self._fps:.0f}")

        delay = self.step_delay_ms if self.auto_mode else 50
        self.root.after(delay, self._update_loop)

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def _update_image_display(self) -> None:
        """Convert current observation to PhotoImage and update the label."""
        if self.obs is None:
            return

        display = self._draw_overlays(self.obs.copy())

        # Scale to fit display area (target ~500px height)
        h, w = display.shape[:2]
        target_h = 500
        scale = target_h / h
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB -> PIL -> ImageTk
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self._tk_image)

    def _draw_overlays(self, img: np.ndarray) -> np.ndarray:
        """Draw trajectory, crosshairs, and action arrow on the image."""
        # Trajectory line (green)
        if len(self.traj_points) > 1:
            for i in range(1, len(self.traj_points)):
                cv2.line(img, self.traj_points[i - 1], self.traj_points[i], (0, 255, 0), 1)

        # Target reference point crosshair (green)
        if self.env is not None and self.env.target_render is not None:
            ref = self.env.target_render.reference_point
            tp = self.info.get("target_position", np.zeros(2))
            mh, mw = self.env.target_render.mask.shape[:2]
            tc = (int(tp[0] + ref[0] - mw / 2), int(tp[1] + ref[1] - mh / 2))
            cv2.drawMarker(img, tc, (0, 255, 0), cv2.MARKER_CROSS, 8, 1)

        # Laser crosshair (red) - image center
        lx, ly = self.sim_config.image_size[1] // 2, self.sim_config.image_size[0] // 2
        cv2.drawMarker(img, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

        # Action arrow (yellow) - from target position
        if self.actions:
            action = self.actions[-1]
            tp = self.info.get("target_position", np.zeros(2))
            start = (int(tp[0]), int(tp[1]))
            end = (int(tp[0] + action[0] * 5), int(tp[1] + action[1] * 5))
            cv2.arrowedLine(img, start, end, (0, 255, 255), 1, tipLength=0.3)

        return img

    # ------------------------------------------------------------------
    # Metrics display
    # ------------------------------------------------------------------

    def _update_metrics_display(self) -> None:
        """Update the right panel with current metrics."""
        dist = self.info.get("distance", 0.0)
        self.dist_label.config(text=f"{dist:.1f} px")
        self.steps_label.config(text=str(self.step_count))

        if self.done:
            is_success = dist < self.sim_config.success_tolerance_px
            self.status_label.config(
                text="SUCCESS" if is_success else "FAILED",
                foreground="green" if is_success else "red",
            )
        else:
            self.status_label.config(text="Running...", foreground="orange")

        # Episode accumulator
        self.ep_count_label.config(text=str(self.episode_total))
        if self.episode_total > 0:
            sr = self.episode_successes / self.episode_total
            self.sr_label.config(text=f"{sr:.1%}")
            avg_d = np.mean(self.episode_distances)
            self.avg_dist_label.config(text=f"{avg_d:.1f} px")
            avg_s = np.mean(self.episode_smoothnesses)
            self.avg_smooth_label.config(text=f"{avg_s:.3f}")

        # Action
        if self.actions:
            a = self.actions[-1]
            self.action_label.config(text=f"dx: {a[0]:+.2f}  dy: {a[1]:+.2f}")
        else:
            self.action_label.config(text="dx: --  dy: --")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self.auto_mode = False
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLA Inference Panel")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG,
                        help="Path to model config YAML")
    parser.add_argument("--sim-config", type=str, default=DEFAULT_SIM_CONFIG,
                        help="Path to simulator config YAML")
    parser.add_argument("--text-cache", type=str, default=DEFAULT_TEXT_CACHE,
                        help="Path to text embeddings cache")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    panel = InferencePanel(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        sim_config_path=args.sim_config,
        text_cache_path=args.text_cache,
        device=device,
    )
    panel.run()


if __name__ == "__main__":
    main()
