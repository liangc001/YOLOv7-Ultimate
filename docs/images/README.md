# 示例图片目录

此目录用于存放 README 中引用的示例图片：

- `training.png` - 训练过程可视化（loss曲线、mAP曲线等）
- `detection.png` - 检测结果示例
- `benchmark.png` - TensorRT 性能对比图

## 如何生成这些图片

### 训练可视化
运行训练后，TensorBoard 日志会保存在 `runs/train/` 目录：
```bash
tensorboard --logdir runs/train
```
截图保存为 training.png

### 检测结果示例
运行检测后，结果会保存在 `runs/detect/` 目录：
```bash
python detect.py --weights yolov7.pt --source data/images
```
截图检测结果保存为 detection.png

### 性能对比
运行 benchmark 后生成对比数据，可以用 matplotlib 绘制对比图：
```python
import matplotlib.pyplot as plt
# 绘制 PyTorch vs TensorRT 速度对比
```
