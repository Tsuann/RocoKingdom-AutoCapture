#!/usr/bin/env python3
"""
对比 RKNN vs ONNX 模型输出，确定问题在转换还是训练。
"""

import sys
import numpy as np
import cv2
import os

print("=" * 60)
print("ONNX 模型输出对比")
print("=" * 60)

# 使用实际画面测试 — 从 HDMI 采集一帧
from capture import HDMIStream

print("\n[1] 采集一帧实际画面...")
stream = HDMIStream(device=0, width=1920, height=1080, fps=30)
if not stream.start():
    print("ERROR: 无法打开 HDMI 采集!")
    sys.exit(1)

import time
frame = None
for i in range(50):
    frame = stream.get_frame()
    if frame is not None:
        print(f"    第 {i+1} 次获取到帧: shape={frame.shape}")
        break
    time.sleep(0.05)

stream.stop()

if frame is None:
    print("ERROR: 无法获取帧!")
    sys.exit(1)

# 保存实际画面
cv2.imwrite("diag_real_frame.png", frame)
print("    已保存: diag_real_frame.png")

# ============================================================
# 1. RKNN 推理
# ============================================================
print("\n[2] RKNN 推理...")
from npu_inference import NPUInference

npu = NPUInference("models/sprite_detector.rknn")

in_w, in_h = 640, 640
f_h, f_w = frame.shape[:2]

scale = min(in_w / f_w, in_h / f_h)
new_w, new_h = int(f_w * scale), int(f_h * scale)
resized = cv2.resize(frame, (new_w, new_h))
letterbox = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
dx = (in_w - new_w) // 2
dy = (in_h - new_h) // 2
letterbox[dy:dy + new_h, dx:dx + new_w] = resized

rknn_input = np.expand_dims(letterbox, axis=0)
rknn_out = npu.run(rknn_input)
print(f"    Output shape: {rknn_out.shape}")
print(f"    Output dtype: {rknn_out.dtype}")
print(f"    Min/Max/Mean: {rknn_out.min():.6f} / {rknn_out.max():.6f} / {rknn_out.mean():.6f}")

# 按 YOLOv8 单类解析
o_rknn = rknn_out.squeeze(0)  # (5, 8400)
if o_rknn.shape[0] < o_rknn.shape[1]:
    o_rknn = o_rknn.T  # (8400, 5)

rknn_bbox = o_rknn[:, :4]
rknn_cls = o_rknn[:, 4]

print(f"    Bbox 列统计: min={rknn_bbox.min():.4f} max={rknn_bbox.max():.4f} mean={rknn_bbox.mean():.4f}")
print(f"    Cls 列统计:  min={rknn_cls.min():.6f} max={rknn_cls.max():.6f} mean={rknn_cls.mean():.6f}")
print(f"    非零 cls 数量: {(rknn_cls != 0).sum()} / {len(rknn_cls)}")

npu.release()

# ============================================================
# 2. ONNX 推理
# ============================================================
print("\n[3] ONNX 推理...")
import onnxruntime as ort

onnx_path = "models/sprite_detector.onnx"
if not os.path.exists(onnx_path):
    print(f"    ONNX model not found: {onnx_path}")
else:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    in_shape = session.get_inputs()[0].shape
    print(f"    ONNX Input:  {input_name} → {in_shape}")

    # ONNX 需要的输入格式: NCHW float32 normalized
    onnx_input = letterbox[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    onnx_input = np.expand_dims(onnx_input, axis=0)
    print(f"    ONNX preprocessed: {onnx_input.shape}, {onnx_input.dtype}")

    onnx_out = session.run(output_names, {input_name: onnx_input})[0]
    print(f"    ONNX Output shape: {onnx_out.shape}")
    print(f"    ONNX Output dtype: {onnx_out.dtype}")
    print(f"    Min/Max/Mean: {onnx_out.min():.6f} / {onnx_out.max():.6f} / {onnx_out.mean():.6f}")

    o_onnx = onnx_out.squeeze(0)  # (C, A)
    if o_onnx.shape[0] < o_onnx.shape[1]:
        o_onnx = o_onnx.T  # (A, C)

    onnx_bbox = o_onnx[:, :4]
    onnx_cls = o_onnx[:, 4]

    print(f"    Bbox 列统计: min={onnx_bbox.min():.4f} max={onnx_bbox.max():.4f} mean={onnx_bbox.mean():.4f}")
    print(f"    Cls 列统计:  min={onnx_cls.min():.6f} max={onnx_cls.max():.6f} mean={onnx_cls.mean():.6f}")
    print(f"    非零 cls 数量: {(onnx_cls != 0).sum()} / {len(onnx_cls)}")

    # Top 10 scores
    top10_idx = np.argsort(onnx_cls)[::-1][:10]
    print(f"\n    ONNX Top-10 class scores:")
    for i, idx in enumerate(top10_idx):
        score = onnx_cls[idx]
        cx, cy, w, h = onnx_bbox[idx]
        print(f"      [{i}] score={score:.6f} cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")

# ============================================================
# 3. 诊断结论
# ============================================================
print(f"\n{'='*60}")
print("诊断结论:")
print(f"{'='*60}")

if onnx_cls is not None and onnx_cls.max() > 0.01:
    print("✓ ONNX 模型有正常的 class scores → 问题在 RKNN 转换")
    print("  可能原因:")
    print("  1. RKNN 量化时 class score 通道被置零")
    print("  2. RKNN 模型导出时 output 配置不对")
    print("  3. RKNN output 的 dequantization 参数 (scale/zp) 有问题")
    print()
    print("  建议: 重新导出 RKNN 模型，检查输出层的 scale 和 zero_point")
elif rknn_cls.max() > 0.01:
    print("✓ RKNN 正常但 ONNX 异常 — 不太可能")
else:
    print("⚠ 两个模型的 class scores 都接近零")
    print("  可能原因:")
    print("  1. 模型训练未收敛 — 分类头没学到东西")
    print("  2. 模型只有 1 个输出通道 (5 通道 = 4 bbox + 1 class)")
    print("     而实际画面中没有可识别目标时 class score 本来就是低值")
    print("  3. 输出格式解析错误")
    print()
    print("  建议:")
    print("  1. 用训练时的验证集测试 ONNX 模型，确认 mAP")
    print("  2. 检查模型训练日志中的 loss 曲线")
    print("  3. 检查导出脚本是否正确保留了所有 8 个类别")
    print(f"  4. 模型 output channels=5 → 只有 1 个类别，不是配置的 8 个!")
