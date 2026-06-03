#!/usr/bin/env python3
"""
修改 ONNX 模型：在 Sigmoid (class scores) 之后乘以 1000，
使其值范围与 bbox 通道 (Mul_2) 相当。
int8 量化后 class score 不再被归零。

ONNX graph:
  Mul_2 → ─┐
            ├─ Concat_3 → output0 (1, 5, 8400)
  Sigmoid → ┘

改为:
  Mul_2 → ─┐
            ├─ Concat_3 → output0
  Sigmoid → Mul(×1000) → ┘

NPU 推理后在 postprocessing 中除以 1000 还原。
"""

import sys, os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ONNX_ORIG = Path("models/sprite_detector.onnx")
ONNX_FIXED = Path("models/sprite_detector_fixed.onnx")
RKNN_OUT = Path("models/sprite_detector.rknn")

CLASS_SCALE = 1000.0

# ============================================================
# 1. Patch ONNX: insert Mul(×1000) after Sigmoid
# ============================================================
print("[1] Patching ONNX model...")
model = onnx.load(str(ONNX_ORIG))
graph = model.graph

# Find Concat_3 node
concat_node = None
for node in graph.node:
    if node.name == '/model.22/Concat_3':
        concat_node = node
        break

if concat_node is None:
    print("ERROR: Cannot find /model.22/Concat_3")
    sys.exit(1)

print(f"    Concat_3 inputs: {concat_node.input}")
# Should be: ['/model.22/Mul_2_output_0', '/model.22/Sigmoid_output_0']

class_input = '/model.22/Sigmoid_output_0'
assert class_input in concat_node.input, f"Sigmoid output not in Concat inputs!"

# Create scale constant — must broadcast with [1, 1, 8400] (rank 3)
# Using shape [1] for scalar broadcast
scale_tensor = helper.make_tensor(
    name='cls_scale_const',
    data_type=TensorProto.FLOAT,
    dims=[1],
    vals=[CLASS_SCALE],
)
graph.initializer.append(scale_tensor)

# Create Mul node after Sigmoid
scaled_class_name = '/model.22/Sigmoid_output_0_scaled'
mul_node = helper.make_node(
    'Mul',
    inputs=[class_input, 'cls_scale_const'],
    outputs=[scaled_class_name],
    name='/model.22/cls_scale_mul',
)
# Update Concat_3 input: replace Sigmoid output with scaled version
for i, inp in enumerate(concat_node.input):
    if inp == class_input:
        concat_node.input[i] = scaled_class_name

# Insert Mul node BEFORE Concat_3 in the node list (topological order)
concat_idx = None
for idx, node in enumerate(graph.node):
    if node.name == '/model.22/Concat_3':
        concat_idx = idx
        break
if concat_idx is not None:
    graph.node.insert(concat_idx, mul_node)
else:
    graph.node.append(mul_node)

print(f"    Inserted Mul(×{CLASS_SCALE}) → {scaled_class_name} (pos {concat_idx})")
print(f"    Concat_3 new inputs: {concat_node.input}")

# Save (skip checker — ONNX Runtime verification below handles it)
onnx.save(model, str(ONNX_FIXED))
print(f"    Saved: {ONNX_FIXED}")

# ============================================================
# 2. Verify
# ============================================================
print("\n[2] Verifying...")
session_orig = ort.InferenceSession(str(ONNX_ORIG), providers=["CPUExecutionProvider"])
session_fixed = ort.InferenceSession(str(ONNX_FIXED), providers=["CPUExecutionProvider"])

test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
input_name = 'images'

out_orig = session_orig.run(None, {input_name: test_input})[0]
out_fixed = session_fixed.run(None, {input_name: test_input})[0]

bbox_diff = np.abs(out_orig[0, :4, :] - out_fixed[0, :4, :]).max()
cls_ratio = out_fixed[0, 4, :].max() / max(out_orig[0, 4, :].max(), 1e-10)

print(f"    Bbox max diff: {bbox_diff:.8f}")
print(f"    Class scale ratio: {cls_ratio:.1f}x (expected {CLASS_SCALE}x)")
print(f"    ✓ VERIFIED" if bbox_diff < 1e-4 and abs(cls_ratio - CLASS_SCALE) < 10 else "    ✗ FAILED!")

# ============================================================
# 3. Convert to RKNN with int8 quantization
# ============================================================
print("\n[3] Converting to RKNN (int8 + calibration)...")
from rknn.api import RKNN

# Calibration dataset (random images — better than nothing)
print("    Creating calibration dataset (30 images)...")
calib_images = []
for i in range(30):
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    calib_images.append(img)
np.save("calib_npy.npy", np.array(calib_images))

rknn = RKNN(verbose=False)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
)

ret = rknn.load_onnx(model=str(ONNX_FIXED))
if ret != 0:
    print(f"    ERROR: load_onnx={ret}")
    sys.exit(1)
print("    ONNX loaded")

ret = rknn.build(do_quantization=True, dataset="calib_npy.npy")
if ret != 0:
    print(f"    ERROR: build={ret}")
    sys.exit(1)
print("    Build complete")

ret = rknn.export_rknn(str(RKNN_OUT))
if ret != 0:
    print(f"    ERROR: export={ret}")
    sys.exit(1)
print(f"    Exported: {RKNN_OUT}")

rknn.release()
print(f"\n{'='*60}")
print(f"DONE! RKNN model: {RKNN_OUT}")
print(f"Class scores are multiplied by {CLASS_SCALE} internally.")
print(f"Postprocessing will divide by {CLASS_SCALE} to restore.")
print(f"{'='*60}")
