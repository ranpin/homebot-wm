# eval/ · 实验产物与复现

本目录存压缩实验的**结果数据**与**产物清单**。大文件（checkpoint / ONNX / 引擎）留在 pc-3070，未入 git（见下方 MANIFEST 的「位置 / 可重建」列）。

## 结果文件（已入库，小件）

| 文件 | 内容 |
|---|---|
| `results/step_sweep_n50.json` | FP32 DDIM 步数曲线 n=50（50/16/5） |
| `results/ddpm100_n50.json` | FP32 DDPM-100 n=50（同 harness 锚点） |
| `results/ddpm50_n50.json` | FP32 DDPM-50 n=50 seed0（崩溃证据） |
| `results/ddpm50_seed1.json` | FP32 DDPM-50 n=50 seed1（崩溃复测） |
| `results/step_sweep_onnx.json` | FP32 DDIM 步数曲线 n=10（**已被 n50 推翻，仅留档**） |
| `results/int8_n50.json` | INT8 闭环 n=50（FP32/A/C 配对对照） |
| `results/orin_latency.json` | **Orin 实测延迟**（trtexec 引擎级 + 全墙钟 sweep + diag 分解；2026-10-19，dog） |

## 复现

所有命令见 [../METHODOLOGY.md](../METHODOLOGY.md) §5。脚本在 [../scripts/](../scripts/)。环境：pc-3070 `dp_venv`（版本见 METHODOLOGY §6）。

```bash
# 在 pc-3070 /data/tmp/chenrunbin/diffusion_policy，export PYTHONPATH=$PWD
# 例：复现 DDIM 步数曲线
./dp_venv/bin/python step_sweep_onnx.py --scheduler ddim --steps 50,16,5 --n_test 50 \
    --checkpoint data/outputs/pusht_img/checkpoints/epoch=0200-test_mean_score=0.876.ckpt
```

## MANIFEST · 产物清单

| 产物 | 大小 | 位置 | 入 git | 可重建 |
|---|---|---|---|---|
| 训练 checkpoint epoch-200 | 4.0 GB | pc-3070 `data/outputs/pusht_img/checkpoints/` | 否 | 否（需重训 ~7h，但训练日志/配置在） |
| ONNX FP32（obs_encoder + unet） | 43 MB + 960 MB | pc-3070 `onnx_export/` | 否 | **是**（`export_onnx.py` + checkpoint，CPU 几分钟） |
| ONNX INT8 per_tensor（A） | 11 MB + 241 MB | pc-3070 `onnx_export_int8/` | 否 | **是**（`quantize_int8.py`，CPU 几分钟） |
| ONNX INT8 per_channel（C） | 11 MB + 278 MB | pc-3070 `onnx_export_int8_pc/` | 否 | **是**（同上 + `--per_channel --op_types Conv`） |
| 结果 JSON | <1 KB 各 | **本目录 `results/`** | **是** | — |
| 评测/训练日志 | 数 MB | pc-3070 `*.log` | 否 | 是（重跑） |
| TensorRT 引擎（FP16+INT8 ×2） | 12.7–506 MB 各 | dog `/home/vision/homebot_wm/engines/` | 否（架构/版本绑定） | **是**（`build_trt_calib.py` + `gen_calib_data.py`，Orin 上几分钟） |

**单点风险**：训练 checkpoint（4GB）只在 pc-3070，丢了需重训 ~7h（8GB GPU）。ONNX/INT8 均可从 checkpoint 重建。TensorRT 引擎在 dog（架构/版本绑定，不入 git），可从 FP32 ONNX + 校准数据用 `build_trt_calib.py` 在 Orin 上重建。

**归档说明**：结果 JSON（小、关键）已入库供公开复现；大文件（checkpoint/ONNX/引擎）gitignore，留 pc-3070，需要时按上表命令重建。SHA256 未逐一记录（产物可重建，非唯一真源）。
