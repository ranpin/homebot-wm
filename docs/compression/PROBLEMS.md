# PROBLEMS · 踩坑记录（现象 → 根因 → 解决 → 教训）

统一问题日志。呈现页 [README.md](README.md) 不含 debug 叙事，全部在此。按阶段分组。

## 阶段 0 · 环境依赖地狱（pc-3070，dp_venv）

Diffusion Policy 是 2023 年仓库，依赖钉死在老版本，与 2026 环境冲突。逐个踩、逐个解：

### P1 · gym 0.21.0 装不上
- **现象**：`pip install gym==0.21.0` 报 `opencv-python>=3.` 非法 requirement。
- **根因**：gym 0.21 的 setup.py 写了尾部带点的非法版本串，现代 pip/wheel 拒绝。
- **解决**：`pip==24.0`（<24.1）+ `wheel==0.38.4` + `setuptools==65.5.0` + `--no-build-isolation`。
- **教训**：老仓库的依赖要连 pip/wheel/setuptools 一起降版本，四件套缺一不可。

### P2 · diffusers / huggingface-hub 版本连锁
- **现象**：装最新 diffusers 后 `hydra.utils.get_class` 报 `Descriptors cannot be created directly`。
- **根因**：仓库用 `diffusers.optimization` + `training_utils.EMAModel`（新版已移除）；diffusers 0.11.1 需 `HfFolder`/`cached_download`（hub 新版已删）。
- **解决**：`diffusers==0.11.1` + `huggingface-hub==0.11.1`。
- **教训**：老仓库 import 的 API 路径是硬约束，装新版必崩。

### P3 · protobuf 被反复拉高
- **现象**：装 robomimic/onnx 后 tensorboard `_pb2` 崩（同 P2 报错）。
- **根因**：robomimic/onnx 把 protobuf 拉到 7.x，tensorboard 的 `_pb2.py` 是旧 protoc 生成。
- **解决**：每次装完 robomimic/onnx 后重 pin `protobuf==3.20.3`。
- **教训**：protobuf 是反复被拉的定时炸弹，装完相关包要复查。

### P4 · robomimic 0.3.0 破坏 CropRandomizer
- **现象**：`module 'robomimic.models.base_nets' has no attribute 'CropRandomizer'`。
- **根因**：robomimic 0.3.0 把 CropRandomizer 移到 `obs_core`，DP 代码引 `base_nets.CropRandomizer`。
- **解决**：`robomimic==0.2.0`。
- **教训**：同 P2，API 位置迁移是硬约束。

### P5 · numpy 被 onnx 强拉到 2.x
- **现象**：装 onnx/onnxruntime 后 torch 1.12 报 `numpy has no attribute 'exceptions'`。
- **根因**：onnx 1.23 强依赖 numpy 2.x；torch 1.12 需 numpy <2。
- **解决**：`onnx==1.14.1` + `onnxruntime==1.15.1` + 重 pin `numpy==1.23.5`。
- **教训**：numpy 是第二个被反复拉的炸弹；onnx 也要选老版。

## 阶段 1 · GPU 硬件故障

### P6 · GPU 掉线，训练死于 epoch 241
- **现象**：`nvidia-smi` → `Unable to determine the device handle for GPU0000:01:00.0: Unknown Error`；训练进程卡死。
- **根因**：GPU 硬件/驱动故障（非软件），恢复需 root（`nvidia-smi --gpu-reset` 或重启），本机无 sudo。
- **解决**：无法本机恢复，需管理员。**但发现压缩全程不需 GPU**（ONNX/INT8/闭环评测全 CPU），主线未被阻塞。
- **教训**：硬件故障要区分「软件可恢复」vs「需 root/换硬件」；同时评估哪些工作可绕开故障硬件继续。

## 阶段 2 · ONNX 导出

### P7 · opset 17 不支持
- **现象**：`ValueError: Unsupported ONNX opset version: 17`。
- **根因**：torch 1.12 最高支持 opset 16。
- **解决**：`--opset 16`。
- **教训**：opset 上限由 torch 版本决定，先查再设。

### P8 · crop_randomizer 的 center_crop 无法 trace
- **现象**：ONNX 导出时 `TypeError: type Tensor doesn't define __round__ method`。
- **根因**：torchvision `center_crop` 对 symbolic shape 做 Python `round()`，tracing 下 shape 是符号张量。
- **解决**：monkey-patch `ttf.center_crop` 用 `torch.narrow`（tensor 索引，可 trace）。
- **教训**：torchvision 的高层 transform 常含 Python 标量运算，ONNX 导出要换成 tensor 原生算子。

## 阶段 3 · INT8 量化

### P9 · per_channel 对 1D 权重崩溃
- **现象**：`per_channel=True` 量化报 `IndexError: tuple index out of range`（`quantize_weight_per_channel`）。
- **根因**：per_channel 沿 channel_axis 量化，但模型有 1D 权重（LayerNorm/GroupNorm/bias），channel_axis 越界。
- **解决**：`--op_types Conv`（只量化 Conv，权重是 3D，per_channel 安全）。
- **教训**：per_channel 不能无脑全开，要限定到有 channel 维的算子。

### P10 · quant_pre_process 符号形状推断失败
- **现象**：`quant_pre_process` 报符号形状推断不完整。
- **根因**：动态 batch 轴导致符号形状推断失败。
- **解决**：`quant_pre_process(skip_symbolic_shape=True, auto_merge=True)`。
- **教训**：动态维度的 ONNX 量化要跳过符号推断。

## 阶段 4 · 评测与结论（方法论教训，最重要）

### P11 · n=10 噪声推翻了 n=10 结论（本项目最大教训）
- **现象**：n=10 时 FP32 ddim16=0.9497，INT8-A 看似掉 8 分、per_channel「救回」。
- **根因**：Push-T 单 episode 覆盖率方差大，n=10 均值噪声带 ±0.05（Wilson）。n=50 复测同配置 FP32 ddim16=0.7379，**差 21 分**。
- **解决**：所有结论性数字改用 n=50。n=50 下 INT8-A=0.788 / INT8-C=0.767 / FP32=0.738，**三者统计不可分 → INT8 免费**，「掉 8 分」是噪声假象。
- **教训**：**小样本单次结果不可信；关键结论必须 n≥50 + 换种子复测。** 反直觉/单次数字先复测再定性（见 reverify-surprising-measurements）。

### P12 · DDPM-50 崩溃的确认
- **现象**：DDPM-50=0.5578（对比 DDIM-50=0.8144），疑似 DDPM 不能减步。
- **根因**：DDPM 反向过程是逐步马尔可夫链，`set_timesteps(50)` 抽稀破坏其数学前提。
- **解决/确认**：换 seed=1 复测得 0.4897，两种子都崩 → **坐实 DDPM 不能减步**，步数压缩必须用 DDIM。
- **教训**：反直觉结果先换种子复测再下结论（用户点名要求复查，见 P11 教训的延伸）。

### P13 · ONNX DDPM-100 < 训练 DDPM-100（跨 harness 差异）
- **现象**：ONNX harness DDPM-100=0.7865 < 训练时 PyTorch DDPM-100=0.876，差 9 分。
- **根因**：跨 harness 差异——env 种子不同（n_envs 50 vs 56）+ DDPM 随机性（不同噪声实现）。**非导出错误**（FP32 parity 1.4e-6 已验证）。
- **解决**：可靠比较只用同 harness 内的对照；跨 harness 的 0.876 仅作参考。
- **教训**：跨 harness/跨 seed 的绝对分数不可直接比；对照必须同 harness 同 seed。

## 阶段 5 · Orin 部署（2026-10-19 闭合）

### P14 · TensorRT/Orin 未实测 → 已闭合
- **现象**：`build_trt.py` 已就绪但无法实测（GPU 故障 + 引擎不跨架构，必须在 Orin 构建）。
- **根因**：TensorRT 引擎架构/版本绑定；本机无 TensorRT 且 GPU 故障。
- **解决（2026-10-19，dog 上闭合）**：TRT 原生 INT8 校准建 4 引擎 + trtexec 引擎级 + ctypes 全墙钟基准，延迟结果见 README 表 4。
- **教训**：引擎类产物不跨架构，构建必须在目标机。

### P15 · 引擎级 cudaGraph 地板 ≠ 全墙钟（一次过度声称的纠错）
- **现象**：早先把 trtexec cudaGraph 引擎和（50×1.96+0.52≈98.5ms）当成部署延迟，报成「DDIM-50+INT8 = 10.1Hz，跨过 10Hz 控制环门槛」。
- **根因**：cudaGraph 引擎和是**理论地板**——假设 CUDA 图捕获 + 零编排开销。真实 Python DDIM 循环用不了 cudaGraph（host 端 numpy 步进打断图捕获），每步实际 ~2.65ms（plain exec 2.43 + Python 0.22）而非 1.96ms，再加 ~21ms 编排开销（h2d/d2h/numpy/encoder）。`bench_predict_action.py` 全墙钟实测 **147ms / 6.8Hz，不过 10Hz**。
- **解决**：ctypes 全墙钟基准 + `--diag` 分解（引擎执行占 93%、编排占 7%，SUM=143ms 闭合）+ `--tight` 紧循环对照（2.67ms/exec ≈ trtexec plain 2.43 + Python）。纠正为「10Hz 门槛只在 DDIM-16（18Hz）跨过，DDIM-50 仅 6.8Hz」。
- **教训**：**部署延迟必须测全墙钟，引擎级 ×N 之和会系统性低估**；cudaGraph 地板只能当理论下界、不能当部署声称。附带：冷启动（warmup 不足）曾把 DDIM-50 虚高到 296ms，是假象非真实——warmup≥5、取 median（dog GPU 与 system_agent 共享，run 间 ±7%）。

### P16 · Push-T 闭环混沌：噪声实现极敏感 + TRT 原生 INT8 闭环掉分 + 单集掩盖系统效应
- **现象 A（噪声敏感）**：闭环 harness 在 dog 上初跑得 ~0 分（FP16 应 ~0.8）。逐项排除引擎（TRT≈onnxruntime 到 1.2e-3）、env（初始态/图像逐像素同 pc-3070）、coverage（块放 goal 位姿 = 1.0）、harness 逻辑后，定位到**初始噪声实现**：我用 numpy `default_rng(0)`，基线用 torch `randn(seed0)`——同策略同 obs，仅换噪声实现，分数从 ~0 到 0.6。
- **根因 A**：Push-T 闭环**混沌 + 有成功阈值**（coverage>0.95）。扩散初始噪声决定轨迹模式；阈值 + 混沌使任何扰动都系统性降分（把「本会成功」的轨迹踢出成功盆容易、反向难）。numpy 噪声恰落在「坏模式」。
- **现象 B（INT8 掉分）**：TRT 原生 INT8 闭环比 FP16 低 ~0.12（6 噪声种子全负，paired t=−6.97，p<0.001），与表 2 的 onnxruntime QDQ INT8「无损」矛盾。
- **根因 B**：TRT 原生熵校准 INT8 的 action 发散（~7-9%，parity 实测）大于 onnxruntime QDQ INT8，该发散被混沌闭环放大成 ~0.12 真实掉分。两条 INT8 路线质量不同。
- **现象 C（单集看不出系统代价）**：为录 Orin 实机 demo 做的 16-seed 扫描（DDIM-50、同一 torch 噪声种子，`eval/results/orin_seedscan_{fp16,int8}.json`）：FP16 均值 0.5433 vs INT8 0.4698（−0.0735），但**逐 seed 差值双向摆动 +0.49 ~ −1.00**——INT8 只在 **6/16** 个 seed 上更低、**7/16 反而更高**、3/16 持平。单看任何一集都可能得出「INT8 更好」或「INT8 完全失效」的相反结论。
- **根因 C**：混沌 + 成功阈值使单集结果由「轨迹是否落进成功盆地」这个近二值事件主导，其放大倍数远大于 INT8 的系统性 action 发散（~7-9%）。系统项（−0.12）只有跨多集平均、把混沌项抵消后才露出来——这正是表 5 必须用 6 噪声种子 × n=50 配对的原因。
- **解决**：① 闭环必须用基线 torch.randn(seed) 噪声（pc-3070 生成 .npy 再 scp）；② INT8-vs-FP16 用 6 噪声种子配对消除噪声实现敏感性；③ 把「INT8 无损」纠正为「onnxruntime QDQ 无损、TRT 原生 INT8 闭环掉 ~0.12」；④ **任何 INT8-vs-FP16 结论只从多集聚合 + 配对统计得出，绝不用单集或少数集举例**；demo 也据此刻意选**双向** seed（10006 打平 1.00/1.00、10011 INT8 退化 0.97/0.59、10002 INT8 反而更好 0.51/1.00）并配说明卡，而非 cherry-pick INT8 失败案例把 −0.12 夸大成「INT8 失效」。
- **教训**：**部署精度必须在目标硬件闭环实测**——引擎级 parity 或 CPU 量化结果不能直接外推；混沌任务对扰动实现极敏感，对照实验必须配对扰动源（噪声种子）。**且在混沌+阈值任务上「单集是噪声、聚合才是信号」**：展示/举证时只挑单集，无论挑哪个方向都会误导。
