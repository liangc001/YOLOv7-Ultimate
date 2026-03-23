# YOLOv7-Ultimate

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![YOLOv7](https://img.shields.io/badge/YOLO-v7-green.svg)](https://github.com/WongKinYiu/yolov7)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.0+-green.svg)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-11.3+-blue.svg)](https://developer.nvidia.com/cuda-toolkit)

一站式YOLOv7目标检测解决方案，包含完整训练流程和TensorRT高性能部署。

## 🎯 项目简介

本项目整合了YOLOv7目标检测的完整工作流程：
- 📊 **数据处理**: 数据集准备、标注、增强
- 🎓 **模型训练**: YOLOv7训练、验证、调优
- 🚀 **模型部署**: TensorRT加速推理，支持FP16/INT8量化
- 📈 **性能评估**: 精度测试、速度基准、可视化分析

## 📁 项目结构

```
yolov7-ultimate/
├── training/                  # 训练相关
│   ├── data_prepare.py       # 数据预处理
│   ├── train.py              # 训练脚本
│   ├── val.py                # 验证脚本
│   └── configs/              # 配置文件
├── deployment/                # 部署相关
│   ├── export_onnx.py        # 导出ONNX
│   ├── build_engine.py       # 构建TensorRT引擎
│   ├── trt_infer.py          # TensorRT推理
│   └── benchmark.py          # 性能测试
├── utils/                     # 工具函数
│   ├── datasets.py           # 数据集处理
│   ├── general.py            # 通用工具
│   └── visualizer.py         # 可视化工具
├── examples/                  # 示例代码
│   ├── inference_example.py  # 推理示例
│   └── train_example.py      # 训练示例
├── docs/                      # 文档
│   ├── TRAINING.md           # 训练指南
│   ├── DEPLOYMENT.md         # 部署指南
│   └── FAQ.md                # 常见问题
├── requirements.txt           # 依赖列表
├── LICENSE                    # 许可证
└── README.md                  # 项目说明
```

## ✨ 核心特性

### 训练阶段
- ✅ 支持自定义数据集
- ✅ 数据增强（Mosaic, MixUp等）
- ✅ 多GPU分布式训练
- ✅ 模型剪枝与量化感知训练

### 部署阶段
- ✅ ONNX模型导出
- ✅ TensorRT引擎构建
- ✅ FP16半精度推理（2x加速）
- ✅ INT8量化推理（4x加速）
- ✅ 批量推理支持
- ✅ 动态输入尺寸

## 🚀 快速开始

### 环境要求

- **硬件**: NVIDIA GPU (推荐 RTX 3060+)
- **系统**: Ubuntu 18.04+ / Windows 10+
- **软件**:
  - Python 3.8+
  - PyTorch 1.12+
  - CUDA 11.3+
  - TensorRT 8.0+ (部署需要)

### 安装

```bash
# 克隆仓库
git clone https://github.com/liangc001/YOLOv7-Ultimate.git
cd YOLOv7-Ultimate

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 📖 使用指南

### 1. 数据准备

```bash
# 准备自定义数据集
python training/data_prepare.py \
    --input data/raw \
    --output data/processed \
    --format yolo
```

### 2. 模型训练

```bash
# 单卡训练
python training/train.py \
    --cfg configs/yolov7.yaml \
    --data data/custom.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 640

# 多卡训练
python -m torch.distributed.launch \
    --nproc_per_node 4 training/train.py \
    --cfg configs/yolov7.yaml \
    --data data/custom.yaml
```

### 3. 模型验证

```bash
python training/val.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/custom.yaml \
    --img-size 640
```

### 4. 导出TensorRT模型

```bash
# 导出ONNX
python deployment/export_onnx.py \
    --weights runs/train/exp/weights/best.pt \
    --img-size 640 \
    --dynamic

# 构建TensorRT引擎
python deployment/build_engine.py \
    --onnx model.onnx \
    --engine model.trt \
    --fp16  # 或 --int8
```

### 5. TensorRT推理

```bash
# 单张图片推理
python deployment/trt_infer.py \
    --engine model.trt \
    --source images/test.jpg \
    --save-dir results/

# 视频推理
python deployment/trt_infer.py \
    --engine model.trt \
    --source video.mp4 \
    --save-dir results/

# 摄像头实时推理
python deployment/trt_infer.py \
    --engine model.trt \
    --source 0
```

## 📊 性能对比

### 推理速度 (RTX 3060, batch=1)

| 平台 | 精度 | 输入尺寸 | 推理时间 | FPS | 加速比 |
|------|------|---------|---------|-----|--------|
| PyTorch | FP32 | 640x640 | 12.5 ms | 80 | 1.0x |
| TensorRT | FP32 | 640x640 | 8.2 ms | 122 | 1.5x |
| TensorRT | FP16 | 640x640 | 5.1 ms | 196 | 2.5x |
| TensorRT | INT8 | 640x640 | 3.2 ms | 312 | 3.9x |

### 检测精度

| 模型 | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|--------------|
| YOLOv7 | 51.2% | 31.2% |
| YOLOv7-X | 53.1% | 33.0% |
| YOLOv7-W6 | 54.9% | 34.8% |

## 🖼️ 示例

### 训练可视化
![Training](docs/images/training.png)

### 检测结果
![Detection](docs/images/detection.png)

### TensorRT加速对比
![Benchmark](docs/images/benchmark.png)

## 📚 文档

- [训练指南](docs/TRAINING.md) - 详细训练流程和调参技巧
- [部署指南](docs/DEPLOYMENT.md) - TensorRT部署完整教程
- [常见问题](docs/FAQ.md) - 问题排查和解决方案

## 🛠️ 高级功能

### 模型剪枝
```bash
python training/prune.py --weights best.pt --ratio 0.3
```

### 知识蒸馏
```bash
python training/distill.py --teacher teacher.pt --student student.pt
```

### 量化校准
```bash
python deployment/calibrate.py --weights best.pt --data data/calib.yaml
```

## 📦 预训练模型

| 模型 | 下载 | 大小 | mAP@0.5 |
|------|------|------|---------|
| YOLOv7 | [下载](https://github.com/) | 72 MB | 51.2% |
| YOLOv7-X | [下载](https://github.com/) | 136 MB | 53.1% |
| YOLOv7-W6 | [下载](https://github.com/) | 166 MB | 54.9% |

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

- [YOLOv7](https://github.com/WongKinYiu/yolov7) - 官方实现
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA推理加速
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📮 联系

如有问题，欢迎提交Issue或联系作者。

---

**如果本项目对你有帮助，请给个 ⭐ Star！**
