# 模型文件目录

此目录存放精灵检测模型文件。

## 模型类型

### RKNN 模型 (推荐, NPU 加速)
- 文件名: `sprite_detector.rknn`
- 推理速度: ~10-30ms (NPU)
- 格式: Rockchip NPU 专用

### ONNX 模型 (CPU 备选)
- 文件名: `sprite_detector.onnx`
- 推理速度: ~100-300ms (CPU)
- 格式: 通用 ONNX

## 如何获取模型

### 方法 1: 自行训练 (当前推荐)

社区模型仓库 `RocoKingdom_AutoCapture` 尚未发布预训练权重（GitHub Releases 为空）。
因此当前推荐的方案是**自行采集数据并训练**。

完整流程:

```bash
# Step 1: 采集游戏截图
# 启动采集工具，在游戏中按 SPACE 保存包含精灵的截图
python3 scripts/capture_screenshots.py

# Step 2: 标注精灵位置
# 打开截图，用鼠标拖拽画出精灵的边界框
python3 scripts/label_tool.py dataset/raw/

# Step 3: 训练模型并导出 ONNX
# # 在 Orange Pi 上（CPU 训练，较慢但已安装依赖）:
python3 scripts/train_model.py --data dataset/labeled/ --epochs 100 --device cpu

# 或在带 GPU 的 PC 上训练（推荐，速度快 50-100 倍）:
# 将 dataset/labeled/ 复制到 PC，然后:
# python3 scripts/train_model.py --data dataset/labeled/ --epochs 100 --device cuda
```

训练完成后，`models/sprite_detector.onnx` 会自动生成。

**数据量建议**: 至少 50-100 张不同场景的截图，每张包含 1-5 个精灵。

### 方法 2: 使用社区预训练模型 (待更新)

从 RocoKingdom_AutoCapture 项目获取:
```
https://github.com/ace-trump-tech/RocoKingdom_AutoCapture
```

> ⚠️ 截至 2026-06-02，该项目尚未在 Releases 中发布模型权重文件。
> 关注该仓库的更新，或联系作者"源批之星·鲁健"获取。
>
> 如果获取到 `s2_all_sprites.pt`，可以用以下命令直接导出 ONNX:
> ```bash
> python3 scripts/train_model.py --export-only --weights s2_all_sprites.pt
> ```

### 方法 3: 无模型运行

即使没有 YOLO 模型，系统也会使用运动检测 + 模板匹配作为后备方案。
准确率较低但可以验证整个流程。

## 模型转换 (PyTorch → RKNN)

需要在 **x86 PC** 上使用 RKNN Toolkit 完整版进行转换:

```bash
# 1. PyTorch → ONNX (可在任意环境完成)
python3 scripts/train_model.py --export-only --weights models/sprite_detector.pt

# 2. ONNX → RKNN (必须在 x86 PC 上运行 RKNN Toolkit)
python3 convert_to_rknn.py
```

转换完成后将 `.rknn` 文件拷贝到此目录。

> 注意: RKNN Toolkit 完整版 (非 lite) 仅支持 x86 Ubuntu/Windows，不能在 ARM 开发板上直接运行。

## 当前状态

如果此目录下没有 `.rknn` 或 `.onnx` 文件，系统会自动使用**运动检测**模式。
虽然精度不如 YOLO，但可以用于测试基本流程。
