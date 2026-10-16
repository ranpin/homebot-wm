# HomeBot-WM

**视觉扩散策略的压缩与边缘部署** —— 面向 Jetson AGX Orin 的 WM / WAM 轻量化全链路学习项目

> 项目目标：学习**世界模型（World Model, WM）/ 世界动作模型（World-Action Model, WAM）**，构建最小模型 demo，打通
> **训练 → 仿真 → 推理优化（压缩）→ 部署**全链路。载体不限，当前为 Push-T。
>
> **当前已交付的主线 = 扩散策略的压缩链路**（train → ONNX → INT8 → DDIM，n=50 验证）。
> 早期的世界模型框架探索（`wm_core` / `wm_sim` / `scripts`）已撞墙并暂停，现状如实见下。

---

## ✅ 已交付主线：扩散策略压缩部署

**[视觉扩散策略的压缩与边缘部署](docs/compression/README.md)** —— 论文式实验报告 · **[在线展示页](https://ranpin.github.io/homebot-wm/)**

载体 = Diffusion Policy（image Push-T 变体，**262M** = 251.5M 扩散 UNet + 11.2M 视觉编码器），在单张 **8GB RTX 3070** 从头训练到 **87.6%**（DDPM-100，n=50），再走完整压缩链路（实验期间 GPU 硬件故障，导出 / 量化 / 闭环评测**全部在 CPU 完成**）：

| 阶段 | 结果 |
|---|---|
| ONNX 导出（拆 `encoder` + `unet` 两组件） | FP32 数值 parity **1.4×10⁻⁶** |
| INT8 静态量化（PTQ，onnxruntime QDQ） | **精度无损**（n=50：FP32 0.738 / INT8 0.767–0.788，统计不可分）；960MB → 241MB |
| 扩散步数压缩 | **DDIM 是唯一可行的减步采样器**（DDPM 抽稀到 50 步崩至 ~0.5，两种子复现）；DDIM-50 = 0.814 ≥ DDPM-100 |
| **推荐部署配置** | **DDIM-50 + INT8**：2× 加速不掉分（0.814 vs 0.787）；极致 DDIM-5 + INT8：20× 加速、−8 分 |

配套：[METHODOLOGY](docs/compression/METHODOLOGY.md)（来源/数据/校准/命令）· [PROBLEMS](docs/compression/PROBLEMS.md)（P1–P14 踩坑）· [scripts](docs/compression/scripts/)（export / quantize / eval / step_sweep / build_trt）· [eval 结果](docs/compression/eval/)

**唯一缺口**：TensorRT / Orin 实机部署 —— `docs/compression/scripts/build_trt.py` 已就绪，因 GPU 硬件故障 + TRT 引擎不跨架构，尚未在 Orin 实测延迟（这是闭环的最后一环）。

**诚实边界**：载体 Push-T 为 2D 仿真（pymunk），精度数字为闭环 `mean_score`（目标区覆盖率 0–1），非真机；「87.6%」不代表真实操作能力。但压缩方法链（ONNX 拆分导出 → QDQ INT8 → DDIM 减步 → TensorRT）与任务无关，可迁移到其他扩散策略 / 扩散世界模型的 Orin 部署。

---

## 🔬 早期探索：世界模型框架（已暂停，如实记录）

`wm_core`（视觉编码 → 潜在动力学 → CEM 规划）+ `wm_sim`（MuJoCo 2D push-block 桌面任务）+ `scripts/` 是最初的世界模型路线。

**结论（详见 [architecture.md](docs/architecture.md) 的 Phase 1 Results）**：

- vision-latent 动力学存在**结构性缺陷（latent collapse）**，闭环成功率 **0%**；
- 只有 state-space MLP + ensemble（disagreement penalty）达到 **25%**（≈ 脚本专家 18%）。

该路线已暂停，项目重心转向上面的压缩部署主线。诊断门（Model Laziness Rate / action sensitivity）保留在 `scripts/diagnose_components.py`。

---

## 项目结构（如实标注）

```
homebot-wm/
├── docs/
│   ├── compression/        # ✅ 主线：压缩部署实验报告 + 5 个流水线脚本 + eval 结果
│   ├── architecture.md     # 世界模型探索的架构与 Phase 1 结果（含诚实的失败记录）
│   └── index.html          # GitHub Pages 展示页（https://ranpin.github.io/homebot-wm/）
├── wm_core/                # 🔬 探索：世界模型核心 encoder / dynamics / planner（已实现）
├── wm_sim/                 # 🔬 探索：MuJoCo 2D push-block 仿真 + 数据集（已实现）
├── scripts/                # 🔬 探索：世界模型训练 / 评估 / 诊断脚本（已实现）
├── configs/                # default.yaml 描述世界模型框架（部分目标未兑现）
├── tests/                  # test_phase_a.py
├── wm_compress/            # ⬜ 占位（仅 __init__.py，未实现；压缩脚本现居 docs/compression/scripts/）
├── wm_deploy/              # ⬜ 占位（未实现；build_trt.py 现居 docs/compression/scripts/）
├── wm_eval/                # ⬜ 占位（未实现）
├── wm_nav/                 # ⬜ 占位（未实现）
└── wm_manip/               # ⬜ 占位（未实现）
```

## 技术栈（如实）

| 环节 | 选型 |
|---|---|
| 训练框架 | PyTorch（压缩载体基于 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)，MIT） |
| 压缩链路 | ONNX / onnxruntime（QDQ INT8 PTQ） |
| 世界模型探索仿真 | MuJoCo（`wm_sim` 2D push-block） |
| 压缩载体评测 | Push-T（pymunk 2D 仿真） |
| 目标部署 | TensorRT / Jetson AGX Orin（待实测） |

## 硬件约束

训练环境为单卡 RTX 3070（8GB VRAM），所有模型设计均在此约束下：

- **仿真与训练分离**：仿真在 CPU 运行，GPU 显存全部留给训练
- **两阶段流程**：先离线采集数据集 → 再从磁盘训练，避免仿真 + 训练共享显存
- **小模型优先**：冻结预训练编码器、LoRA 微调、AMP FP16、gradient checkpointing
- **目标部署**：NVIDIA Jetson AGX Orin

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> 注：压缩主线的训练在外部 diffusion_policy 仓库完成，本仓库 `docs/compression/scripts/` 承载导出 / 量化 / 评测 / 部署脚本；世界模型探索的训练/评估脚本在 `scripts/`。

## License

MIT
