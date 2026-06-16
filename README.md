# HomeBot-WM

**World Model-driven Lightweight Algorithm Framework for Home Robotics**

基于世界模型的家庭机器人轻量化算法框架，面向 NVIDIA Jetson AGX Orin 部署，结合扩散模型剪枝/量化/蒸馏优化经验。

## 项目特点

- **世界模型驱动**：使用扩散型世界模型（Diffusion Policy）统一导航与操作任务
- **轻量化部署**：结构化剪枝 → 知识蒸馏 → TensorRT 量化，实现 Orin 实时推理
- **仿真优先**：基于 Isaac Sim 构建家庭场景仿真环境，支持 Sim2Real 迁移
- **SD 优化经验复用**：将 Stable Diffusion 压缩优化技术迁移到世界模型领域

## 技术架构

```
┌─────────────────────────────────────────────┐
│           Application Layer                  │
│  导航任务 │ 操作任务 │ 复合任务（移动抓取）    │
├─────────────────────────────────────────────┤
│           World Model Core                   │
│  视觉编码器 │ 动态预测器 │ 动作规划器          │
├─────────────────────────────────────────────┤
│         Lightweight Engine                    │
│  模型压缩 │ TensorRT 部署 │ 推理调度           │
└─────────────────────────────────────────────┘
```

## 项目结构

```
homebot-wm/
├── wm_core/          # 世界模型核心（视觉编码、动态预测、动作规划）
├── wm_compress/      # 模型压缩工具链（剪枝、量化、蒸馏）
├── wm_deploy/        # Orin 部署引擎（ONNX 导出、TensorRT、推理流水线）
├── wm_nav/           # 导航模块
├── wm_manip/         # 操作模块
├── wm_sim/           # 仿真环境（Isaac Sim）
├── wm_eval/          # 评估框架
├── configs/          # 配置文件
├── scripts/          # 训练/部署脚本
├── notebooks/        # 实验 notebook
├── examples/         # 示例代码
└── docs/             # 文档
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 深度学习框架 | PyTorch 2.x |
| 仿真环境 | Isaac Sim |
| 机器人中间件 | ROS 2 Humble |
| 部署引擎 | TensorRT |
| 模型格式 | ONNX → TRT Engine |
| 语言 | Python（研究）+ C++（部署）|

## 开发环境

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e ".[dev]"
```

## 里程碑

- **Phase 0** — 基础建设：项目骨架 + 开发环境 + 仿真搭建
- **Phase 1** — 世界模型 MVP：Diffusion Policy baseline + 仿真验证
- **Phase 2** — 轻量化优化：剪枝 + 量化 + 蒸馏
- **Phase 3** — Orin 部署：TensorRT 推理 + 性能调优
- **Phase 4** — 综合验证：导航+操作复合任务 + Sim2Real

## License

MIT
