# Architecture

## Overview

HomeBot-WM is a world model-driven algorithm framework for home robotics, targeting deployment on NVIDIA Jetson AGX Orin. The framework combines navigation and manipulation under a unified world model architecture, with a strong emphasis on lightweight deployment through model compression techniques.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────────┐     │
│   │  wm_nav  │  │ wm_manip │  │  Composite Tasks      │     │
│   │  (导航)  │  │ (操作)   │  │  (移动抓取/跟随等)    │     │
│   └────┬─────┘  └────┬─────┘  └──────────┬───────────┘     │
│        └──────────────┼───────────────────┘                  │
├───────────────────────┼─────────────────────────────────────┤
│                  World Model Core (wm_core)                  │
│                                                              │
│   ┌────────────┐  ┌────────────┐  ┌────────────────┐       │
│   │  Visual    │  │  Dynamics  │  │   Action       │       │
│   │  Encoder   │→ │  Predictor │→ │   Planner      │       │
│   │  (ViT/CNN) │  │ (Diffusion)│  │ (CEM/MPC)      │       │
│   └────────────┘  └────────────┘  └────────────────┘       │
│         ↑                ↑                ↓                  │
│     RGB-D Input     State+Action    Action Sequence          │
├─────────────────────────────────────────────────────────────┤
│              Lightweight Engine (wm_compress + wm_deploy)    │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐         │
│   │ Pruning  │  │   QAT    │  │   TensorRT       │         │
│   │(结构化)  │→ │(量化感知) │→ │  Engine Build    │         │
│   └──────────┘  └──────────┘  └──────────────────┘         │
│         ↑                                ↓                   │
│   Teacher Model                   FP16/INT8 Inference        │
│   Distillation                    Pipeline Scheduler         │
└─────────────────────────────────────────────────────────────┘
         ↕                                    ↕
┌──────────────────┐              ┌──────────────────────┐
│     wm_sim       │              │      wm_eval         │
│  MuJoCo (CPU)    │              │  Nav/Manip/Deploy    │
│  家庭场景仿真    │              │  Metrics & Reports   │
└──────────────────┘              └──────────────────────┘
```

## Module Responsibilities

### wm_core — World Model Core

The central module that learns environment dynamics.

- **encoder/**: Visual encoder (frozen ResNet18 backbone + lightweight adapter) that processes RGB-D observations into latent representations. Encoder is frozen to stay within 8GB VRAM training budget; only adapter layers are trained.
- **dynamics/**: Diffusion-based dynamics model that predicts future states given current observation and proposed action sequence. Inspired by Diffusion Policy and UniSim.
- **planner/**: Action planner that uses the dynamics model to evaluate candidate action sequences (CEM, MPC, or shooting methods) and select the optimal one.

### wm_compress — Compression Toolkit

Leverages Stable Diffusion optimization experience for world model compression.

- **pruning/**: Structured pruning (channel/layer removal) with importance-based criteria. Targets 50%+ parameter reduction while preserving task performance.
- **quantization/**: Quantization-aware training (QAT) and post-training quantization (PTQ) for INT8/FP16 inference.
- **distillation/**: Teacher-student knowledge distillation. Train a compact student model guided by the full-size teacher.
- **benchmark/**: Automated evaluation comparing compressed vs. original models across task metrics and inference speed.

### wm_deploy — Orin Deployment

End-to-end deployment pipeline for Jetson AGX Orin.

- **export/**: PyTorch → ONNX conversion with dynamic batch support.
- **tensorrt/**: TensorRT engine building with FP16/INT8 precision, layer fusion, and memory optimization.
- **pipeline/**: Multi-model inference scheduler that orchestrates perception → world model → planning with minimal latency.

### wm_nav — Navigation

- World model-driven path planning with collision prediction
- Local obstacle avoidance using dynamics model lookahead
- Global map maintenance and goal-directed navigation

### wm_manip — Manipulation

- Grasp planning with world model outcome prediction
- 6-DoF manipulation policy generation via diffusion model
- Manipulation result verification through dynamics prediction

### wm_sim — Simulation

- MuJoCo-based home environment (kitchen, living room, bedroom) — runs on CPU to preserve GPU VRAM for training
- Two-phase data pipeline: offline collection → static HDF5 datasets → training from disk
- Domain randomization for Sim2Real transfer
- Automated data collection with scripted/expert policies

**Future iteration — Isaac Sim + offline data**: When GPU resources permit (multi-GPU or cloud), migrate to Isaac Sim for higher-fidelity rendering and physics. The data collection pipeline remains the same — collect trajectories offline, store to disk, train separately. This decouples simulation fidelity from training VRAM budget.

### wm_eval — Evaluation

- **Navigation**: success rate, path efficiency, collision rate
- **Manipulation**: grasp success rate, task completion rate
- **Deployment**: inference latency (ms), throughput (FPS), GPU utilization

## Data Flow

```
Camera (RGB-D)
    ↓
Visual Encoder → latent state
    ↓
Dynamics Model (current_state, candidate_actions) → predicted_states
    ↓
Planner (predicted_states, goal) → optimal_action
    ↓
Robot Controller → execute action
```

## Design Principles

1. **Simulation-first**: All development and validation starts in MuJoCo (CPU) before real hardware.
2. **VRAM-aware**: All model designs target the 8GB VRAM training budget (RTX 3070). Simulation runs on CPU; training uses mixed precision, frozen encoders, and gradient checkpointing.
3. **Compression-aware**: Models are designed with deployment constraints in mind from day one.
4. **Modular**: Each module can be developed, tested, and replaced independently.
5. **Progressive**: Start with full-precision models, then compress iteratively.

## VRAM Budget (Training: RTX 3070 8GB)

| Component | Strategy | Estimated VRAM |
|-----------|----------|---------------|
| Visual encoder | Frozen ResNet18, no grad | ~0.1 GB |
| Dynamics model (Diffusion) | hidden_dim=128, 3 layers, AMP FP16 | ~1.5 GB |
| Planner | CEM, no backprop | ~0.2 GB |
| Optimizer states (AdamW) | Only adapter + dynamics params | ~1.0 GB |
| Activations + gradients | Gradient checkpointing, batch_size=8 | ~3.0 GB |
| MuJoCo sim | CPU only | 0 GB |
| **Total** | | **~5.8 GB** |

Batch size kept at 8 with gradient accumulation (effective batch = 64) to leave headroom.
