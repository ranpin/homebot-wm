# METHODOLOGY · 来源 / 数据 / 校准 / 指标 / 命令

呈现页见 [README.md](README.md)。本文档只放可复现细节，所有版本号实测取自环境（非记忆）。

## 1. 模型与权重来源

| 项 | 值 | 来源 |
|---|---|---|
| 载体架构 | Diffusion Policy, image Push-T 变体 | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)（MIT），`git clone --depth 1`，**未 pin commit**（可复现性缺口，建议 pin） |
| Workspace 类 | `TrainDiffusionUnetHybridWorkspace` | 仓库 `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py` |
| 配置 | `image_pusht_diffusion_policy_cnn.yaml` | 仓库根（下载自项目实验目录） |
| 权重 | **我们自己从头训练**，epoch-200 | `data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt`（pc-3070） |
| 扩散 UNet | `ConditionalUnet1D`，251.5M 参数，down_dims→2048 | 实测自训练日志 `Diffusion params: 2.515119e+08` |
| 视觉编码器 | `MultiImageObsEncoder`，11.2M 参数 | 实测自训练日志 `Vision params: 1.119709e+07` |
| 调度器 | `DDPMScheduler`，num_train_timesteps=100，epsilon 预测，squaredcos_cap_v2 | 配置实测 |

**关键超参**（实测自 config）：horizon=16，n_obs_steps=2，n_action_steps=8，action_dim=2，image=(3,96,96)，agent_pos=(2,)，global_cond=132（=2×66），num_inference_steps=100（默认 DDPM）。

**训练**：8GB RTX 3070，batch_size=64，~50s/epoch，epoch-200 时 test_mean_score=0.876（DDPM-100，训练内置 eval，n_test=50）。loss 0.846→0.436（epoch 1→20，未完全收敛但已平台）。

## 2. 数据集（分清三类）

| 类别 | 数据 | 说明 |
|---|---|---|
| 训练集 | `data/pusht/pusht_cchi_v7_replay.zarr` | Push-T 官方 replay buffer，**非我们采集**。img (25650,96,96,3) float32 [0,255]、state (25650,5)、action (25650,2)、keypoint (25650,9,2)，206 episodes。下载自 `diffusion-policy.cs.columbia.edu`（~48 kB/s） |
| 评测集 | `PushTImageEnv`（pymunk 仿真） | 闭环 rollout，**无固定数据集**，env 按 seed 生成初始状态。n_test=50 episodes |
| 量化校准集 | 从训练 zarr 采 128 个真实观测 | image [0,1]（=zarr img/255）+ agent_pos（=state[:,:2] 像素坐标）。UNet 校准输入：global_cond 由 encoder 对真实观测编码 + sample=randn + timestep~U[0,100) |

**obs 格式**（实测自 `pusht_image_dataset.py`）：`image = moveaxis(img,-1,1)/255` → [0,1]；`agent_pos = state[:,:2]`（像素坐标，未归一化，由 normalizer 处理）。normalizer：image 用 range normalizer，agent_pos 用 limits-fit。

## 3. 量化 / 校准

| 项 | 值 |
|---|---|
| 方法 | onnxruntime 静态 PTQ（`quantize_static`），QDQ 格式 |
| 校准样本 | 128（真实观测） |
| 权重类型 | QInt8 |
| 配置 A | per_channel=False，op_types=全部（默认） |
| 配置 C | per_channel=True，op_types=Conv（per_channel 对 1D 权重会崩，故限定 Conv） |
| 预处理 | `quant_pre_process(skip_symbolic_shape=True, auto_merge=True)`（动态 batch 轴导致符号形状推断失败） |

**校准的已知局限**：UNet 校准用 randn sample + 均匀 timestep，覆盖高噪声区；更精确应按 timestep 用 forward-diffuse 的真实动作（`scheduler.add_noise`）。当前配置下 INT8 已精度无损，未进一步精修。

## 4. 指标定义

- **mean_score**：Push-T 闭环成功率指标 = T 形块对目标区的**覆盖率**（0–1，连续），n 个 episode 取均值。**非二值成功**。由 `PushTImageRunner.run()` 返回 `test/mean_score`。
- **parity（max_abs_diff）**：ONNX 输出 vs PyTorch 输出的最大绝对差，`np.allclose(atol=1e-4)` 判过。
- **INT8-vs-FP32 diff**：同输入下 INT8 ONNX 与 FP32 ONNX 输出的最大绝对差（单步，非闭环）。
- **UNet 调用/预测**：每次 `predict_action` 的 UNet 前向次数 = num_inference_steps（视觉编码器恒 1×）。

**绝对 vs 相对口径**：mean_score 是绝对覆盖率；「−8 分」是相对 DDPM-100 基线的绝对差。同 harness（ONNX CPU）内的对照才可比；训练时 PyTorch eval（0.876）是跨 harness 参考，不直接比。

## 5. 确切命令

所有命令在 pc-3070 `/data/tmp/chenrunbin/diffusion_policy`，`export PYTHONPATH=$PWD`，venv `./dp_venv/bin/python`。

```bash
# 训练载体模型（8GB GPU）
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0 training.num_epochs=500 \
    logging.mode=disabled hydra.run.dir=data/outputs/pusht_img

# ONNX 导出（CPU 即可）+ parity 校验
python export_onnx.py --checkpoint <ckpt> --out_dir onnx_export --opset 16

# INT8 PTQ（CPU）——配置 A（per_tensor 全 op）/ 配置 C（per_channel 仅 Conv）
python quantize_int8.py --checkpoint <ckpt> --onnx_dir onnx_export \
    --out_dir onnx_export_int8 --n_calib 128                       # A
python quantize_int8.py --checkpoint <ckpt> --onnx_dir onnx_export \
    --out_dir onnx_export_int8_pc --n_calib 128 --per_channel --op_types Conv   # C

# 闭环评测（CPU，pymunk + onnxruntime）——FP32 vs INT8，同步数同种子
python eval_onnx_pusht.py --checkpoint <ckpt> --int8_dir onnx_export_int8_pc \
    --steps 16 --n_test 50 --skip_fp32

# 步数 / 采样器扫描（CPU）
python step_sweep_onnx.py --checkpoint <ckpt> --scheduler ddim --steps 50,16,5 --n_test 50
python step_sweep_onnx.py --checkpoint <ckpt> --scheduler ddpm --steps 50 --n_test 50 --seed 1

# TensorRT 引擎构建（必须在 Orin 上跑，引擎不跨架构）—— 待执行
python build_trt.py --onnx_dir onnx_export --int8_dir onnx_export_int8_pc \
    --out_dir trt_engines --precisions fp16,int8 --benchmark
```

## 6. 环境与版本（实测）

**硬件**：pc-3070 = NVIDIA RTX 3070 8GB（sm_86），16 核，**15GB RAM**，CUDA 12.4 驱动 550.144.03。
**⚠️ GPU 于 2026-10-10 硬件故障**（`nvidia-smi` → `Unknown Error`），训练死于 epoch 241；**压缩全程在 CPU 完成**（ONNX 导出 / INT8 / 闭环评测均不需 GPU）。恢复需 root（无 sudo）。

**软件**（`dp_venv`，Python 3.10）：

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 1.12.1+cu116 | cu116 在 CUDA 12.4 驱动上向后兼容 |
| numpy | 1.23.5 | torch 1.12 需 <2；scipy 1.15 需 ≥1.23.5 |
| gym | 0.21.0 | 需 pip<24.1 + wheel 0.38.4 + setuptools 65.5.0 + --no-build-isolation |
| diffusers | 0.11.1 | 仓库用 `diffusers.optimization`（新版已移除） |
| huggingface-hub | 0.11.1 | diffusers 0.11.1 需 `HfFolder`/`cached_download`（新版已删） |
| protobuf | 3.20.3 | robomimic/onnx 会拉到 7.x → tensorboard `_pb2` 崩 |
| robomimic | 0.2.0 | 0.3.0 把 CropRandomizer 移到 obs_core → DP 代码崩 |
| onnx / onnxruntime | 1.14.1 / 1.15.1 | 新版强拉 numpy 2.x |
| numba | 0.56.4 | sampler 依赖 |

完整依赖踩坑见 [PROBLEMS.md](PROBLEMS.md)。

## 7. 复现验证状态

| 数字 | 复现次数 | 状态 |
|---|---|---|
| ONNX parity 1.4e-6 | 1 | 确定性，已验 |
| INT8 无损（n=50） | 1（FP32/A/C 同轮） | 同种子配对 |
| DDIM 步数曲线（n=50） | 1 | 同种子 |
| DDPM-50 崩溃 | **2 种子**（0.558/0.490） | 已复测确认 |
| n=10 vs n=50 差异 | 2 | 已确认 n=10 不可信 |

**run-to-run 方差**：DDPM-50 两种子差 ~7 分（0.558 vs 0.490），反映坏采样器的高方差；DDIM/INT8 各配置为确定性（同种子复现到小数点后 4 位，已用 ddim16 验证 harness 确定性）。
