# VLA：基于扩散策略的视觉-语言-动作显微目标导航系统

一个完整的 **视觉-语言引导精细显微目标定位** 框架，采用条件扩散策略（Conditional Diffusion Policy）实现。系统学习将显微镜视野中的微小目标（微球、酵母细胞、精子头部、精子尾部）自动导航至激光焦点位置，支持通过域随机化与少量真实演示实现从模拟器到真实 Nikon Ti2E 显微镜的迁移部署。

## 项目概述

本项目实现了一个基于 **Diffusion Policy** 的 VLA（Vision-Language-Action）模型：

1. 输入显微镜图像 + 语言指令
2. 预测平滑的多步动作序列（载物台像素空间位移）
3. 通过闭环控制逐步将目标导航至激光点
4. 通过域随机化 + 少样本微调实现从仿真到真实硬件的迁移

该框架用统一架构同时处理 **刚性物体**（微球）和 **可变形生物样本**（酵母细胞、精子）。

## 项目结构

```
VLA/
├── configs/                        # YAML 配置文件
│   ├── model/
│   │   └── diffusion_policy.yaml   # 模型超参数
│   ├── simulator/
│   │   ├── default.yaml            # 默认模拟器设置
│   │   ├── microsphere.yaml        # 微球任务配置
│   │   ├── yeast.yaml              # 酵母任务配置
│   │   ├── sperm_head.yaml         # 精子头部任务配置
│   │   └── sperm_tail.yaml         # 精子尾部任务配置
│   └── training/
│       └── default.yaml            # 训练超参数
├── data/
│   ├── checkpoints/                # 保存的模型权重 (.pt)
│   ├── runs/                       # TensorBoard 日志
│   ├── trajectories/               # 生成的专家演示轨迹
│   │   ├── microsphere/
│   │   ├── yeast/
│   │   ├── sperm_head/
│   │   └── sperm_tail/
│   └── text_embeddings.pt          # 缓存的 CLIP 嵌入
├── scripts/
│   ├── generate_data.py            # 专家轨迹生成
│   ├── train.py                    # 训练入口
│   ├── eval.py                     # 批量评估
│   └── run_inference_panel.py      # 启动交互式测试 GUI
├── src/
│   ├── vla/                        # 核心 VLA 模型
│   │   ├── diffusion_policy.py     # 完整扩散策略（训练 + 推理）
│   │   ├── diffusion_unet.py       # 1D U-Net 噪声预测网络
│   │   ├── visual_encoder.py       # ResNet-18 视觉骨干网络
│   │   ├── text_encoder.py         # CLIP 文本编码器 + 缓存
│   │   ├── condition_embed.py      # 视觉-文本融合 MLP
│   │   ├── noise_scheduler.py      # DDPM/DDIM 噪声调度器
│   │   └── inference.py            # 闭环推理控制器
│   ├── simulator/                  # 显微镜模拟器
│   │   ├── environment.py          # Gym 风格显微镜环境
│   │   ├── renderer.py             # 图像合成渲染管线
│   │   ├── targets.py              # 参数化目标生成器
│   │   ├── stage.py                # 模拟 Nikon Ti2E 载物台
│   │   ├── background.py           # Perlin 噪声 / 图像背景
│   │   ├── noise.py                # 传感器噪声模型
│   │   └── expert_generator.py     # 基于 PID 的专家演示生成
│   ├── data/
│   │   ├── dataset.py              # 滑动窗口轨迹数据集
│   │   └── augmentation.py         # 翻转、亮度、噪声数据增强
│   ├── training/
│   │   ├── trainer.py              # 训练循环 + TensorBoard
│   │   ├── evaluator.py            # 批量策略评估
│   │   └── metrics.py              # 成功率、距离、平滑度指标
│   ├── ui/
│   │   ├── inference_panel.py      # Tkinter 交互式测试面板
│   │   └── simulator_viewer.py     # OpenCV 模拟器可视化
│   ├── interfaces/
│   │   └── stage_interface.py      # 载物台抽象接口（仿真 ↔ 真实）
│   └── utils/
│       ├── config.py               # 数据类配置 + YAML 加载器
│       ├── logger.py               # 日志设置
│       └── perlin.py               # Perlin 噪声生成器
├── tests/                          # 单元测试
├── doc/
│   └── paper-design.md             # 论文设计文档
├── requirements.txt
└── README_cn.md                    # 中文说明文档（本文件）
```

## 安装

### 环境要求

- Python 3.10+
- 支持 CUDA 的 GPU（推荐）或 CPU

### 安装步骤

```bash
# 克隆仓库
git clone <your-repo-url>
cd VLA

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate        # Linux/Mac
# 或: venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

| 包名 | 用途 |
|------|------|
| `torch` / `torchvision` | 模型训练与推理 |
| `transformers` | CLIP 文本编码器（仅预计算时使用） |
| `opencv-python` | 图像渲染与处理 |
| `numpy` | 数值计算 |
| `pyyaml` | 配置文件加载 |
| `tensorboard` | 训练过程可视化 |
| `tqdm` | 进度条显示 |
| `Pillow` | GUI 图像显示 |
| `pytest` | 单元测试 |

## 快速开始

### 第一步：生成专家演示数据

模拟器使用 PID 控制器自动生成专家轨迹：

```bash
# 生成单个任务的数据
python scripts/generate_data.py --task microsphere --num-trajectories 5000

# 生成所有任务的数据（microsphere, yeast, sperm_head, sperm_tail）
python scripts/generate_data.py --all-tasks --num-trajectories 5000

# 使用多进程并行生成
python scripts/generate_data.py --all-tasks --num-trajectories 5000 --workers 4
```

每条轨迹保存为一个文件夹，包含：
- `images/` — JPEG 帧序列（224×224）
- `actions.npy` — (T, 2) 的 (dx, dy) 像素位移数组
- `positions.npy` — (T+1, 2) 的目标位置数组
- `meta.json` — 轨迹元数据

### 第二步：预计算文本嵌入

CLIP 嵌入只需计算一次并缓存：

```bash
python scripts/train.py --precompute-text --tasks all
```

此命令创建 `data/text_embeddings.pt`，包含每个任务指令的 512 维 CLIP 嵌入：
- `"Navigate the microsphere to the red laser dot."`
- `"Navigate the yeast cell to the red laser dot."`
- `"Navigate the sperm head to the red laser dot."`
- `"Navigate the sperm tail tip to the red laser dot."`

### 第三步：训练扩散策略模型

```bash
# 在所有任务上训练
python scripts/train.py --model diffusion_policy --tasks all

# 在指定任务上训练
python scripts/train.py --model diffusion_policy --tasks microsphere,yeast

# 从检查点恢复训练
python scripts/train.py --model diffusion_policy --tasks all --resume data/checkpoints/epoch_0100.pt

# 使用 GPU
python scripts/train.py --model diffusion_policy --tasks all --device cuda

# 使用自定义配置
python scripts/train.py --model diffusion_policy --tasks all \
    --model-config configs/model/diffusion_policy.yaml \
    --training-config configs/training/default.yaml
```

训练输出：
- `data/checkpoints/best.pt` — 验证损失最优的检查点
- `data/checkpoints/epoch_NNNN.pt` — 周期性检查点
- `data/runs/<timestamp>/` — TensorBoard 日志

实时监控训练过程：
```bash
tensorboard --logdir data/runs
```

### 第四步：评估训练好的模型

```bash
# 在单个任务上评估（100 个 episode）
python scripts/eval.py --checkpoint data/checkpoints/best.pt --task microsphere --episodes 100

# 在所有任务上评估
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100

# 保存结果到 JSON 文件
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100 --output results.json

# 使用 GPU 加速推理
python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --device cuda
```

输出指标：
- **成功率（Success Rate）** — 最终距离 < 2 像素的 episode 比例
- **平均最终距离（Mean Final Distance）** — episode 结束时目标到激光点的平均像素距离
- **轨迹平滑度（Trajectory Smoothness）** — 相邻动作向量之间的平均角度变化（越低越平滑）

### 第五步：使用 GUI 进行交互式测试

```bash
# 启动推理面板
python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt

# 使用 GPU
python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt --device cuda
```

GUI 功能说明：
- **左侧控制面板**：任务选择器、噪声/域随机化开关、速度控制、单步/自动按钮
- **中央显示区**：实时显微镜视野，叠加显示轨迹线（绿色）、目标十字（绿色）、激光十字（红色）、动作箭头（黄色）
- **右侧面板**：实时指标（距离、步数、成功/失败）、episode 汇总（成功率、平均距离、平均平滑度）、预测动作 (dx, dy)
- **键盘快捷键**：`Space/S` = 单步执行，`A` = 自动运行，`R` = 重置环境，`1-4` = 切换任务，`Q/Esc` = 退出

## 方法详解

### 整体架构

模型采用 **条件扩散策略** 架构，由四个核心模块组成：

```
                    ┌─────────────────┐
                    │   语言指令       │
                    │  (CLIP 编码)     │
                    └────────┬────────┘
                             │ text_emb (512维)
┌──────────────┐             │
│  观测历史     │             │
│  (2帧图像)    │             │
└──────┬───────┘             │
       │                      │
       ▼                      │
┌──────────────┐             │
│ 视觉编码器    │             │
│ (ResNet-18)   │             │
│ → 256维特征   │             │
└──────┬───────┘             │
       │ visual_feat          │
       ▼                      ▼
┌──────────────────────────────┐
│     条件嵌入模块              │
│     拼接 → MLP → 256维       │
└──────────────┬───────────────┘
               │ 条件向量 c
               ▼
┌──────────────────────────────┐
│     1D 扩散 U-Net            │
│                              │
│  x_T ~ N(0,I) ──→ 逐步去噪  │
│  t (时间步)                   │
│  c (条件向量)                 │
│                              │
│  每个下采样/上采样块          │
│  通过 FiLM 注入条件           │
│                              │
│  → 预测噪声                  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     DDIM 采样器               │
│     10步去噪（训练: 100步）   │
│     → 动作序列 (10步 × 2维)  │
└──────────────────────────────┘
```

#### 1. 视觉编码器（`src/vla/visual_encoder.py`）

- **骨干网络**：ResNet-18（ImageNet 预训练）
- **输入修改**：第一层卷积修改为接受 `obs_horizon × 3` 个通道（2帧 RGB 拼接 = 6通道）。预训练权重复制到新增通道。
- **输出**：256 维特征向量，经过 `Linear → ReLU → LayerNorm`
- **观测窗口**：2 帧（提供时序上下文，隐式估计目标运动方向）

#### 2. 文本编码器（`src/vla/text_encoder.py`）

- **模型**：CLIP ViT-B/32（`openai/clip-vit-base-patch32`）
- **用法**：CLIP 仅在训练前使用一次，为每个任务的固定语言指令预计算 512 维嵌入。训练和推理时 **不加载** CLIP 模型。
- **缓存**：嵌入保存到 `data/text_embeddings.pt`，以简单字典形式加载。

#### 3. 条件嵌入模块（`src/vla/condition_embed.py`）

将视觉和文本特征融合为单一条件向量：

```
concat(视觉_256维, 文本_512维) → Linear(768→512) → ReLU → Dropout(0.1)
                              → Linear(512→256) → ReLU → Dropout(0.1)
                              → Linear(256→256) → LayerNorm
```

输出：256 维条件向量 `c`，通过 FiLM 调制驱动扩散 U-Net。

#### 4. 扩散 U-Net（`src/vla/diffusion_unet.py`）

一个 **1D 卷积 U-Net**，从带噪动作序列中预测噪声：

- **输入**：带噪动作 `(B, 2, 10)` + 时间步 `t` + 条件向量 `c`
- **输出**：预测噪声 `(B, 2, 10)`
- **网络结构**：
  - **时间步嵌入**：正弦位置编码 → MLP → 256 维
  - **条件融合**：`[时间步嵌入; 条件向量]` → Linear → 256 维 FiLM 向量
  - **编码器**：`Conv1d(2→64) → DownBlock(64→128) → DownBlock(128→256)`
  - **瓶颈层**：两层 Conv1d + FiLM 条件注入
  - **解码器**：`UpBlock(256+128→128) → UpBlock(128+64→64)`，带跳跃连接
  - **输出层**：`Conv1d(64→2)`，插值到目标序列长度
- **FiLM 条件注入**：在每个块中，条件向量生成缩放和偏移参数：`output = x × (1 + scale) + shift`
- **DownBlock 结构**：`Conv1d(stride=2) → GroupNorm → ReLU → Conv1d → GroupNorm → FiLM → ReLU + 残差连接`
- **UpBlock 结构**：`Upsample(2×) → cat(跳跃连接) → Conv1d → GroupNorm → ReLU → Conv1d → GroupNorm → FiLM → ReLU + 残差连接`

#### 5. 噪声调度器（`src/vla/noise_scheduler.py`）

- **训练调度**：100 步扩散，余弦 beta 调度
- **推理调度**：DDIM 10 步（比 DDPM 快 10 倍）
- **前向扩散**：`x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε`
- **DDIM 反向去噪**：从 `x_t` 和噪声预测推断 `x_0`，再计算 `x_{t-1}`，过程是确定性的

### 训练流程

**损失函数**：标准去噪得分匹配 — 预测噪声与实际噪声的均方误差：

```python
# 伪代码
visual_feat = visual_encoder(obs_seq)           # (B, 256)
cond = condition_embed(visual_feat, text_emb)   # (B, 256)
noise = randn(B, 2, 10)                         # 随机噪声
t = randint(0, 100)                             # 随机时间步
noisy_actions = scheduler.add_noise(actions, noise, t)
noise_pred = unet(noisy_actions, t, cond)
loss = MSE(noise_pred, noise)
```

**优化器**：AdamW（学习率=1e-4，权重衰减=1e-6）

**学习率调度**：线性预热（10 个 epoch）+ 余弦衰减（总计 500 个 epoch）

**数据增强**（`src/data/augmentation.py`）：
- 随机水平翻转（概率 0.5）— 同步翻转图像和 x 方向动作
- 随机亮度/对比度抖动
- 高斯噪声注入

**训练技巧**：
- 梯度累积支持
- 基于验证损失的早停（耐心值=100）
- 每 50 个 epoch 周期性保存检查点
- TensorBoard 日志记录（损失、学习率）

### 推理流程（闭环控制）

测试时，模型以 **闭环** 方式运行：

```python
obs_history = [env.reset()]
while not done:
    obs_seq = pad_and_stack(obs_history[-2:])  # 取最近2帧
    action_seq = model.predict_action(obs_seq, text_emb)  # DDIM 10步去噪
    action = action_seq[0]  # 只取第一个动作
    obs = env.step(action)
    obs_history.append(obs)
```

**核心设计**：模型预测 10 步动作序列，但 **只执行第一步**。下一帧到来时重新预测。这种 **滚动时域（receding horizon）** 方法提供了对累积误差的鲁棒性。

### 模拟器设计（`src/simulator/`）

模拟器渲染合成显微镜图像，完全控制视觉外观：

**渲染管线**（`renderer.py`）：
1. 生成 Perlin 噪声背景（或加载真实图像裁剪）
2. 在指定位置粘贴目标物体 mask（alpha 混合）
3. 在图像中心绘制红色激光点（半径=3 像素）
4. 应用域随机化（亮度、对比度、高斯模糊）
5. 应用传感器噪声（高斯噪声、椒盐噪声、光照闪烁、运动模糊）

**目标生成器**（`targets.py`）：

| 目标 | 形状 | 尺寸 | 参考点 |
|------|------|------|--------|
| 微球（Microsphere） | 圆形（实心或环形） | 半径=15-25 像素 | 圆心 |
| 酵母（Yeast） | 椭圆 + 正弦凸起 | 长轴=20-35 像素 | 中心 |
| 精子头（Sperm Head） | 椭圆 + 锥形尖端 | 长轴=18-28 像素 | 中心 |
| 精子尾（Sperm Tail） | 三次 Bezier 曲线 + 小头部 | 长度=40-100 像素 | 尾尖端 |

**载物台模拟**（`stage.py`）：
- 镜像真实 Nikon Ti2E 载物台 API（抽象接口定义在 `interfaces/stage_interface.py`）
- 以微米为单位跟踪位置，范围可配置
- 像素到微米的转换带有每 episode ±10% 的随机标定误差

**专家演示生成**（`expert_generator.py`）：
- PID 控制器将目标参考点导航至激光点
- 添加随机动作噪声模拟人类操作员抖动
- 动作幅度限制在 30 像素/步以内
- 每条轨迹记录：图像（JPEG）、动作（npy）、位置（npy）、元数据（json）

### 评估指标（`src/training/metrics.py`）

| 指标 | 定义 |
|------|------|
| **成功率（Success Rate）** | 最终距离 < 2 像素的 episode 比例 |
| **平均最终距离（Mean Final Distance）** | episode 结束时目标到激光点的平均欧氏距离（像素） |
| **轨迹平滑度（Trajectory Smoothness）** | 相邻动作向量之间的平均绝对角度变化（范围 [0, π]，越低越平滑） |
| **轨迹长度（Trajectory Length）** | 完成 episode 所需的步数 |

## 配置说明

### 模型配置（`configs/model/diffusion_policy.yaml`）

```yaml
obs_horizon: 2              # 观测帧数
pred_horizon: 10            # 动作序列长度
action_dim: 2               # (dx, dy) 位移
visual_backbone: "resnet18"
visual_output_dim: 256
text_dim: 512               # CLIP 嵌入维度
condition_hidden_dims: [512, 256]
condition_output_dim: 256
unet_dims: [64, 128, 256]   # U-Net 通道数递进
num_diffusion_steps: 100    # 训练扩散步数
num_ddim_steps: 10          # DDIM 推理步数
beta_schedule: "cosine"
```

### 训练配置（`configs/training/default.yaml`）

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

### 模拟器配置（`configs/simulator/default.yaml`）

```yaml
image_size: [224, 224]
um_per_pixel: 0.5
um_per_pixel_noise: 0.1        # ±10% 标定误差
max_steps_per_episode: 200
success_tolerance_px: 2.0
start_distance_min: 50.0       # 起始距激光点最小距离
start_distance_max: 200.0      # 起始距激光点最大距离

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

## 运行测试

```bash
# 运行所有单元测试
pytest tests/

# 运行特定测试文件
pytest tests/test_simulator.py
pytest tests/test_vla.py
```

## Sim-to-Real 迁移策略

框架设计为两阶段部署到真实显微镜：

**阶段 1 — 模拟器训练**：在完整域随机化条件下（背景、光照、噪声、标定误差），使用每个任务 5000+ 条生成轨迹进行训练。

**阶段 2 — 真实世界微调**：在 Nikon Ti2E 显微镜上采集 10-20 条真实演示，然后以低学习率对预训练模型微调 50-100 个 epoch。可以选择冻结视觉编码器，只微调 U-Net 动作头。

`StageInterface` 抽象接口（`src/interfaces/stage_interface.py`）允许在环境构造函数中用一行代码将 `StageSimulator` 替换为真实的 `NikonTi2EStage` 实现。

## 核心技术亮点

| 特性 | 实现方式 |
|------|----------|
| 动作生成 | Diffusion Policy (DDPM/DDIM)，天然支持多模态、平滑轨迹 |
| 视觉编码 | ResNet-18，2帧拼通道提供时序运动信息 |
| 语言条件 | CLIP 预计算缓存，训练时零额外开销 |
| 条件注入 | FiLM 调制，在 UNet 每个块注入条件信息 |
| 推理加速 | DDIM 10步采样（训练用100步） |
| 闭环控制 | 只执行预测序列第一步，每帧重新预测 |
| 专家数据 | PID 控制器 + 人手抖动噪声模拟 |
| 域随机化 | 背景 + 形状 + 光照 + 噪声 + 标定误差 |
| 接口抽象 | StageInterface 统一仿真/真实 API |

## 许可证

[待填写]

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{your_citation,
  title={Vision-Language-Action Diffusion Policy for Micro-Object Navigation},
  author={Your Name},
  year={2026}
}
```
