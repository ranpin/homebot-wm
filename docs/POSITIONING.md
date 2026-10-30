# homebot-wm 项目定位与诚实边界

README 讲核心结果与快速开始；本文讲项目定位（命名由来、原目标 vs 交付主线）与诚实边界清单。

## 命名与定位：原目标 vs 交付主线

仓库名的「WM」（World Model）是项目起点：原目标为学习**世界模型（WM）/ 世界动作模型（WAM）**，构建最小模型 demo，打通**训练 → 仿真 → 推理优化（压缩）→ 部署**全链路（载体不限，选定 Push-T）。

实际交付的主线 = **视觉扩散策略（Diffusion Policy）的压缩与边缘部署**（train → ONNX → INT8 → DDIM 减步 → TensorRT / Orin 实机实测，n=50 验证）。注意 **Diffusion Policy 不是世界模型**——它是视觉模仿学习（预测 p(A|obs)，不预测未来状态）；世界模型预测 p(z'|z,a) 供规划。「WM」留在仓库名里是历史痕迹。

价值定位：**边缘部署方法论的全链路**（训练 → 压缩 → 部署 → 实机实测；压缩方法链构造上任务无关）；机器人操作能力本身不是贡献点（Push-T 为 2D 仿真、无真机）。

## 早期 WM 框架探索：撞墙与暂停

`wm_core`（视觉编码 → 潜在动力学 → CEM 规划）+ `wm_sim`（MuJoCo 2D push-block）+ `scripts/` 是最初的世界模型路线。结论（详见 [architecture.md](architecture.md) 的 Phase 1 Results）：

- vision-latent 动力学存在**结构性缺陷（latent collapse）**，闭环成功率 **0%**——diffusion 动力学比 identity 基线更差（0.051 vs 0.005）、residual-MLP 仅打平（0.0051 vs 0.0053），都撑不起 CEM 规划；
- 只有 state-space residual MLP + ensemble（5×，disagreement penalty λ=1.0）达 **25%**（单模型 ~17.5%，≈ 脚本专家 18%）——但它吃 **6 维真实状态** `[agent_xy, agent_vxy, block_xy]`（真机不可得）、且只有几 M 参数（无可压缩对象），偏离「压缩 + 部署视觉模型」主线。

该路线已暂停；诊断门（Model Laziness Rate / action sensitivity）保留在 `scripts/diagnose_components.py`。

## 诚实边界清单

- 载体 Push-T 为 **2D 仿真（pymunk）、非真机**；精度数字为闭环 `mean_score`（目标区覆盖率 0–1，取回合内最大值）。
- 「87.6%」是**训练内置 eval**，不代表真实操作能力；同模型 ONNX harness 复测 **0.787**。
- 压缩方法链（ONNX 拆分导出 → QDQ INT8 → DDIM 减步 → TensorRT）**构造上**与任务无关（各环节不含 Push-T 专用逻辑），但「可迁移」是**设计推断、未在第二个任务上实证**。
- 「INT8 无损」只对 onnxruntime QDQ 路线成立；**TRT 原生熵校准 INT8 在 Orin 闭环掉 ~0.12**（6 噪声种子 × n=50 配对，t=−6.97，p<0.001），不能外推。
- 压缩阶段 GPU 硬件故障，导出 / 量化 / 闭环评测全部在 **CPU** 完成；部署阶段在 Orin 上闭合。
