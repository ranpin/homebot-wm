# HomeBot-WM

> 把视觉扩散策略（Diffusion Policy）压缩并部署到 Jetson AGX Orin 的全链路学习项目：8GB 单卡训练 → ONNX 导出 → INT8 量化 → DDIM 减步 → TensorRT 实机实测。项目起点是世界模型（WM）学习——WM 框架探索已撞墙暂停、交付主线为压缩部署，仓库名由此带有历史痕迹（定位与诚实边界见 [docs/POSITIONING.md](docs/POSITIONING.md)）。

> 📊 **在线报告 / 展示页**：<https://ranpin.github.io/homebot-wm/>

## 核心结果

**[视觉扩散策略的压缩与边缘部署](docs/compression/README.md)** —— 论文式实验报告。

载体 = Diffusion Policy（image Push-T 变体，**262M** = 251.5M 扩散 UNet + 11.2M 视觉编码器；输入 2 帧观测——96×96 RGB 图像 [0,1] + agent 2D 像素坐标，输出 16 步 2D 动作序列、执行前 8 步后重规划），在单张 **8GB RTX 3070** 从头训练到 **87.6%**（DDPM-100，训练内置 eval），再走完整压缩链路并**部署到 Jetson AGX Orin 实机实测**（压缩阶段 GPU 硬件故障，导出 / 量化 / 闭环评测全部在 CPU 完成；部署阶段在 Orin 上闭合）：

| 阶段 | 结果 |
|---|---|
| ONNX 导出（拆 `encoder` + `unet` 两组件） | FP32 数值 parity **1.4×10⁻⁶** |
| INT8 静态量化（PTQ，onnxruntime QDQ） | n=50 与 FP32 **统计不可分**（FP32 0.738 / INT8 0.767–0.788）；960MB → 241MB |
| 扩散步数压缩 | **DDIM 是唯一可行的减步采样器**（DDPM 抽稀到 50 步崩至 ~0.5，两种子复现）；DDIM-50 = 0.814 与 DDPM-100 = 0.787 **统计不可分**（一半算力、精度不降） |
| **TensorRT / Orin 实机部署** | 4 引擎建成（TRT 10.7 原生 INT8 熵校准）；**全 `predict_action` 墙钟实测**：DDIM-16 FP16 **83ms / 12Hz**、INT8 **55ms / 18Hz**；DDIM-50 INT8 147ms / 6.8Hz |
| **Orin 闭环精度实测** | TRT 原生 INT8 比 FP16 **掉 ~0.12**（6 噪声种子 × n=50 配对，t=−6.97，**p<0.001**）——「INT8 无损」只对 onnxruntime QDQ 路线成立，不能外推到 TRT |
| **推荐部署配置** | 精度优先 **DDIM-16 + FP16**（12Hz）；速度/内存优先 **DDIM-16 + INT8**（18Hz，牺牲 ~0.12 精度换 1.6× 提速 + ~4× 压缩） |

> 口径注：Push-T 为 2D 仿真（pymunk）、非真机；精度 = 闭环 `mean_score`（目标区覆盖率 0–1，取回合内最大值）；「87.6%」是训练内置 eval，同模型 ONNX harness 复测 0.787。完整诚实边界清单见 [docs/POSITIONING.md](docs/POSITIONING.md)。

配套：[METHODOLOGY](docs/compression/METHODOLOGY.md)（来源/数据/校准/指标/命令/环境版本）· [PROBLEMS](docs/compression/PROBLEMS.md)（P1–P16 踩坑）· [scripts](docs/compression/scripts/)（12 个：导出 / 量化 / 评测 / 步数扫描 / TRT 构建与校准 / Orin 延迟与闭环基准 / demo 制作）· [eval 结果](docs/compression/eval/)

**闭环状态**：训练 → 压缩 → 部署 → **实机实测（延迟 + 闭环精度）已全部打通**，无遗留缺口。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

注：压缩主线的训练在外部 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) 仓库（@ `5ba07ac`）完成，本仓 `docs/compression/scripts/` 承载导出 / 量化 / 评测 / 部署脚本（确切复现命令见 [METHODOLOGY](docs/compression/METHODOLOGY.md)）；早期 WM 探索的训练/评估脚本在 `scripts/`。

## 仓库结构与文档导航

```
homebot-wm/
├── docs/
│   ├── compression/        # ✅ 主线：压缩部署实验报告（README/METHODOLOGY/PROBLEMS）+ 12 个脚本 + eval 结果 + demo 视频
│   ├── POSITIONING.md      # 项目定位与诚实边界（命名由来 / WM 探索暂停结论 / 边界清单）
│   ├── architecture.md     # 世界模型探索的架构与 Phase 1 结果（含诚实的失败记录）
│   └── index.html          # GitHub Pages 展示页（https://ranpin.github.io/homebot-wm/）
├── wm_core/                # 🔬 早期探索：世界模型核心 encoder / dynamics / planner（已暂停）
├── wm_sim/                 # 🔬 早期探索：MuJoCo 2D push-block 仿真 + 数据集
├── scripts/                # 🔬 早期探索：WM 训练 / 评估 / 诊断脚本（diagnose_components.py 保留诊断门）
├── configs/                # default.yaml 描述早期 WM 框架（部分目标未兑现）
├── tests/                  # test_phase_a.py
├── wm_compress/            # ⬜ 占位（仅 __init__.py，未实现；压缩脚本现居 docs/compression/scripts/）
├── wm_deploy/              # ⬜ 占位（未实现；build_trt.py 现居 docs/compression/scripts/）
├── wm_eval/                # ⬜ 占位（未实现）
├── wm_nav/                 # ⬜ 占位（未实现）
└── wm_manip/               # ⬜ 占位（未实现）
```

早期 WM 探索结论（如实）：vision-latent 动力学存在**结构性缺陷（latent collapse）**，闭环成功率 **0%**；只有 state-space MLP + ensemble（disagreement penalty）达 **25%**（≈ 脚本专家 18%），但它吃 6 维真实状态（真机不可得）且仅几 M 参数（无可压缩对象），偏离主线——该路线已暂停，详见 [docs/architecture.md](docs/architecture.md) 的 Phase 1 Results 与 [docs/POSITIONING.md](docs/POSITIONING.md)。

## 技术栈与硬件约束

| 环节 | 选型 |
|---|---|
| 训练框架 | PyTorch（压缩载体基于 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) @ `5ba07ac`，MIT） |
| 压缩链路 | ONNX / onnxruntime（QDQ INT8 PTQ）+ TensorRT 10.7（原生熵校准 INT8 / FP16） |
| 世界模型探索仿真 | MuJoCo（`wm_sim` 2D push-block） |
| 压缩载体评测 | Push-T（pymunk 2D 仿真） |
| 部署目标 | Jetson AGX Orin（**已实测**：4 引擎 + 全墙钟延迟 + 闭环精度） |

训练环境为单卡 RTX 3070（8GB VRAM），所有模型设计均在此约束下：**仿真与训练分离**（仿真在 CPU 运行，GPU 显存全部留给训练）、**两阶段流程**（先离线采集数据集 → 再从磁盘训练）、**小模型优先**（冻结预训练编码器、LoRA 微调、AMP FP16、gradient checkpointing）。

## 许可与致谢

MIT——见 [`LICENSE`](LICENSE)。压缩载体基于 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)（MIT，@ `5ba07ac`）；Push-T 评测环境与训练配方来自该仓库。
