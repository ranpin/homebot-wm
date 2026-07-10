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
│   │  (ResNet)  │  │(Diff./MLP) │  │ (CEM/MPC)      │       │
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

- **encoder/**: Visual encoder (frozen ResNet18 backbone + lightweight adapter) that processes RGB observations into latent representations. Encoder is frozen to stay within 8GB VRAM training budget; only adapter layers are trained. **Note:** images must be normalized to `[0, 1]` (divide by 255) at both training and inference — the frozen backbone is sensitive to input scale.
- **dynamics/**: Latent dynamics model predicting the next latent from `(latent, action)`. Two interchangeable backends are selected at load time via `build_dynamics(config)` on `config['dynamics_type']`:
  - `diffusion` (`DiffusionDynamics`): DDPM model, full or respaced-DDIM sampling. Inspired by Diffusion Policy / UniSim.
  - `mlp` (`MLPDynamics`): deterministic **residual** regressor (`next = latent + MLP(latent, action)`), zero-initialized so it starts at the identity baseline. Added because, on the near-static tabletop task, the diffusion backend's per-step sampling variance exceeded the true frame-to-frame latent change and scored *worse than identity* — see the diagnostic below.
- **planner/**: Action planner that uses the dynamics model to evaluate candidate action sequences (CEM, MPC, or shooting methods) and select the optimal one. `plan(latent, score_fn, num_steps=None)` forwards `num_steps` to the dynamics rollout.

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

#### Current MVP task & dataset

- **Task** (`HomeTabletopEnv`, `assets/home_tabletop.xml`): top-down 2D *push-block-to-target*. A point-mass agent pushes a block into a fixed target zone at `[1.5, 1.5]` amid 3 fixed obstacles in a walled arena. Observation: 84×84 overhead RGB + 6-D state `[agent_xy, agent_vxy, block_xy]`. Action: 2-D force in `[-1, 1]`. Success: `dist(block, target) < 1.0` within 200 steps. Only the agent/block start positions are randomized; target and obstacles are fixed.
- **Dataset** (`data/trajectories.h5`, ~235 MB): 2000 episodes / 360,647 transitions collected by a scripted potential-field expert (`wm_sim/expert.py`). Stores image/state/action/reward/success per step.
- **Data-quality caveat**: the scripted expert only succeeds **~18%** of the time (most episodes time out). This is weak for *imitation*, but acceptable for learning *dynamics* — failed rollouts still contain valid physics transitions (the block decoder reaches ~0.35 error). The main risk is thin state coverage near the target; if closed-loop planning stalls on the "last push", prefer success-weighted resampling or an improved expert over changing the task.

### wm_eval — Evaluation

- **Navigation**: success rate, path efficiency, collision rate
- **Manipulation**: grasp success rate, task completion rate
- **Deployment**: inference latency (ms), throughput (FPS), GPU utilization

**Component diagnostic** (`scripts/diagnose_components.py`, CPU-friendly): validates each learned component against dataset ground truth without a full closed-loop run — block-decoder position error, encoder input-normalization sensitivity, and dynamics 1-step rollout MSE **vs. an identity ("no-move") baseline**. A dynamics model is only usable for planning if its rollout MSE beats identity; this is the acceptance gate before closed-loop evaluation.

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

## Phase 1 Results (push-block MVP)

Closed-loop success on the push-block task, comparing dynamics backends (CEM planner,
frozen encoder where applicable). Acceptance gate: dynamics rollout must beat the
identity "no-move" baseline **and** produce a usable planning signal.

| Approach | 1-step rollout vs identity | Closed-loop success |
|----------|----------------------------|---------------------|
| Diffusion latent dynamics | worse (0.051 vs 0.005) | 0% |
| Residual-MLP latent dynamics | ~tie (0.0051 vs 0.0053) | 0% |
| Residual-MLP state-space dynamics (single) | beats (0.0004 vs 0.0016) | ~17.5% (stable, 40 ep) |
| **Ensemble (5×) state-space + disagreement penalty** | — | **25%** ✅ (λ=1.0, 40 ep) |

Key findings:
- The original eval fed `[0,255]` images to an encoder trained on `[0,1]` — a
  normalization mismatch that alone forced 0% (decoder error 96 vs 0.35). Fixed.
- **Vision-latent 1-step dynamics carries too little action signal**: in most
  transitions the block is static (the agent is still navigating), so "no move"
  is near-optimal and the action barely matters at 1 step → CEM cannot plan.
- **State-space world model works**: a residual MLP over the 6-D true state
  `[agent_xy, agent_vxy, block_xy]` learns real push dynamics (block-position
  prediction 0.007 vs 0.019 identity); CEM then plans successful pushes at
  roughly the level of the scripted expert that generated the data (~18%).
- Success is currently **~17–18% (stable over 40 episodes), i.e. expert level**.
- Single-model CEM shows **model exploitation**: longer horizons / more samples /
  running cost all made it *worse* (5–8%) — hard optimization against one
  imperfect model finds actions that look good to the model but fail in reality.
- **Ensemble with disagreement penalty fixes this** (`evaluate_state_ensemble.py`):
  5 members trained from different seeds; each candidate is scored
  `mean(score) − λ·std(score)` so CEM avoids sequences the members disagree on.
  λ=1.0 → **25%** (vs 17.5% single-model); λ=0 (plain averaging, no penalty) →
  12.5%, confirming it is the *penalty* that helps, not ensembling alone; λ=2.0
  (over-conservative) → 17.5%.
- Further headroom needs better data (target-region coverage) or a multi-step /
  vision-latent world model — the latter remains the harder research track.

**Demo**: `python scripts/evaluate_state.py --checkpoint checkpoints/state_dynamics.pt --record_dir demo`
saves a GIF of each successful episode (overhead view: red agent pushes the blue
block into the green target zone, avoiding brown obstacles).

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
| Dynamics model (Diffusion or residual MLP) | hidden_dim=128–256, 3 layers, AMP FP16 | ~1.5 GB |
| Planner | CEM, no backprop | ~0.2 GB |
| Optimizer states (AdamW) | Only adapter + dynamics params | ~1.0 GB |
| Activations + gradients | Gradient checkpointing, batch_size=8 | ~3.0 GB |
| MuJoCo sim | CPU only | 0 GB |
| **Total** | | **~5.8 GB** |

Batch size kept at 8 with gradient accumulation (effective batch = 64) to leave headroom.
