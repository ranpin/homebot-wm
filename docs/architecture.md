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
│  Isaac Sim       │              │  Nav/Manip/Deploy    │
│  家庭场景仿真    │              │  Metrics & Reports   │
└──────────────────┘              └──────────────────────┘
```

## Module Responsibilities

### wm_core — World Model Core

The central module that learns environment dynamics.

- **encoder/**: Visual encoder (ViT or CNN backbone) that processes RGB-D observations into latent representations. Pre-trained on large-scale datasets and fine-tuned for home environments.
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

- Isaac Sim-based home environment (kitchen, living room, bedroom)
- Domain randomization for Sim2Real transfer
- Automated data collection and annotation pipeline

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

1. **Simulation-first**: All development and validation starts in Isaac Sim before real hardware.
2. **Compression-aware**: Models are designed with deployment constraints in mind from day one.
3. **Modular**: Each module can be developed, tested, and replaced independently.
4. **Progressive**: Start with full-precision models, then compress iteratively.
