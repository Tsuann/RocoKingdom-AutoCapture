# 模型文件目录

此目录存放精灵检测模型文件。

## 当前模型状态

| 模型文件 | 状态 | 说明 |
|---------|------|------|
| `sprite_detector.pt` | ✅ V4 | PyTorch 权重，小独角兽 (mAP50=0.994)，6.2MB |
| `sprite_detector.onnx` | ✅ V4 | ONNX 模型，部署用，11.7MB |
| `sprite_detector.rknn` | ❌ 待转换 | 需在 x86 PC 用 RKNN Toolkit 转换 |

### 已训练类别

| ID | 精灵 | 数据 | 夜间 | 白天 |
|----|------|------|------|------|
| 6 | 小独角兽 | 2段视频 + 10张夜间 + 9张白天 | 10/10 | 9/10 |

### 待采集

| ID | 精灵 | 状态 |
|----|------|------|
| 0 | 护主犬 | 待录像 |
| 1 | 伊贝儿 | 待录像 |
| 2 | 恶魔叮 | 待录像 |
| 3 | 菊花梨 | 待录像 |
| 4 | 公平鸽 | 待录像 |
| 5 | 灵狐 | 待录像 |
| 7 | 小夜/朔夜伊芙 | 待录像 |

## 模型类型

### ONNX 模型 (开发板部署)
- 文件名: `sprite_detector.onnx`
- 推理速度: ~100-300ms (Orange Pi CPU) / ~1ms (PC GPU)
- 格式: 通用 ONNX (opset 12, simplified)

### RKNN 模型 (NPU 加速, 待转换)
- 文件名: `sprite_detector.rknn`
- 推理速度: ~10-30ms (NPU)
- 需用 RKNN Toolkit 从 ONNX 转换

## PC 训练流程

```bash
# 1. 录视频 — OBS 框选精灵区域，2-5 分钟多角度旋转
# 2. 提取帧
python scripts/extract_frames.py dataset/video/xxx.mp4 --sprite xiao_dujiaoshou --fps 2
# 3. 自动标注
python scripts/auto_label.py dataset/raw_xiao_dujiaoshou/ --class-id 6
# 4. 训练 (RTX 5060 Ti ~10分钟)
python scripts/incremental_train.py --new-data dataset/labeled_xiao_dujiaoshou/ --device cuda
# 5. ONNX 自动导出到 models/sprite_detector.onnx

# 部署到开发板
scp models/sprite_detector.onnx orangepi@<ip>:/home/orangepi/Desktop/rock/models/
```

## 无模型运行

如果 `models/` 下没有 `.onnx` 文件，系统自动使用运动检测作为后备方案。
