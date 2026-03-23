# YOLOv7-Ultimate

## 项目说明

本项目合并自：
- yolov7_targettest (训练测试)
- yolov7_tensorrt_beta (TensorRT部署)

整合为一个完整的目标检测解决方案。

## 原项目位置

如需查看原始代码，请访问：
- https://github.com/liangc001/yolov7_targettest (已归档)
- https://github.com/liangc001/yolov7_tensorrt_beta (已归档)

## 新架构

```
yolov7-ultimate/
├── training/      # 原 yolov7_targettest 内容
├── deployment/    # 原 yolov7_tensorrt_beta 内容
├── utils/         # 共享工具函数
├── examples/      # 示例代码
└── docs/          # 文档
```

## 迁移说明

原项目代码将按功能分类移动到对应目录：
- 训练相关 → training/
- 推理部署 → deployment/
- 工具函数 → utils/

## 时间线

- 2023: 创建 yolov7_targettest
- 2023: 创建 yolov7_tensorrt_beta
- 2024: 合并为 YOLOv7-Ultimate
