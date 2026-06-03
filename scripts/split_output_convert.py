#!/usr/bin/env python3
"""
修改 ONNX 模型：将单输出 (1,5,8400) 拆分为双输出 (1,4,8400) + (1,1,8400)
bbox 和 class score 各自独立量化，解决 class score 通道被归零的问题。

然后转换为 RKNN 模型（int8 量化，带校准数据集）。

用法: python3 scripts/split_output_convert.py
"""

import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
ONNX_ORIG = MODEL_DIR / "sprite_detector.onnx"
ONNX_SPLIT = MODEL_DIR / "sprite_detector_split.onnx"
RKNN_OUT = MODEL_DIR / "sprite_detector.rknn"


def split_onnx_output():
    """在 ONNX 图中添加 Split 节点，将单输出分成 bbox 和 class 两个输出。"""
    import onnx
    from onnx import helper, TensorProto

    print(f"[1] Loading ONNX: {ONNX_ORIG}")
    model = onnx.load(str(ONNX_ORIG))

    # 找到原始输出节点
    graph = model.graph
    output_name = None

    # 查找原始输出
    for node in graph.node:
        if node.op_type == "Concat" and len(node.output) == 1:
            out = node.output[0]
            # 检查是否连接到 graph output
            for g_out in graph.output:
                if g_out.name == out:
                    output_name = out
                    break

    if output_name is None:
        # 找最后一个节点
        for g_out in graph.output:
            output_name = g_out.name

    print(f"   原始输出名: {output_name}")

    # 查找产生此输出的节点
    last_node = None
    for node in graph.node:
        for out in node.output:
            if out == output_name:
                last_node = node
                break
        if last_node:
            break

    if last_node is None:
        print("ERROR: 找不到输出节点!")
        return False

    print(f"   最后一个节点: {last_node.op_type} ({last_node.name})")

    # 创建 Split 节点：沿 axis=1 (channel) 分割为 [0:4] 和 [4:5]
    split_node = helper.make_node(
        'Split',
        inputs=[output_name],
        outputs=['bbox_output', 'cls_output'],
        name='output_split',
        axis=1,
    )

    # 更新 graph：替换输出节点
    # 1) 移除所有使用 output_name 的 graph output
    new_outputs = [o for o in graph.output if o.name != output_name]
    # 2) 移除产生 output_name 的节点（如果不是输入节点）
    #    实际上我们保留它，只是把它的输出改个名字
    #    更简单的方法：在 Concat 后面添加 Split
    graph.node.append(split_node)

    # 3) 添加新的 graph outputs
    bbox_out = helper.make_tensor_value_info('bbox_output', TensorProto.FLOAT, [1, 4, 8400])
    cls_out = helper.make_tensor_value_info('cls_output', TensorProto.FLOAT, [1, 1, 8400])
    new_outputs.extend([bbox_out, cls_out])
    graph.output.clear()
    graph.output.extend(new_outputs)

    # 保存
    onnx.save(model, str(ONNX_SPLIT))
    print(f"   Split ONNX saved: {ONNX_SPLIT}")

    # 验证
    import onnxruntime as ort
    session = ort.InferenceSession(str(ONNX_SPLIT), providers=["CPUExecutionProvider"])
    for inp in session.get_inputs():
        print(f"   Input:  {inp.name} {inp.shape}")
    for out in session.get_outputs():
        print(f"   Output: {out.name} {out.shape}")

    return True


def calibrate_and_convert():
    """用校准数据集转换 split ONNX → RKNN (int8)。"""
    from rknn.api import RKNN

    # 采集校准图片
    print(f"\n[2] Preparing calibration dataset...")
    calib_dir = PROJECT_ROOT / "dataset" / "images" / "train"
    calib_files = list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png"))
    if not calib_files:
        calib_files = list((PROJECT_ROOT / "dataset").rglob("*.jpg"))
        calib_files += list((PROJECT_ROOT / "dataset").rglob("*.png"))

    # 如果没有训练图片，生成一些随机噪声作为校准数据
    if len(calib_files) < 10:
        print(f"   WARNING: 仅有 {len(calib_files)} 张校准图片，使用合成数据...")
        calib_images = []
        for i in range(50):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            calib_images.append(img)
        calib_dataset = "calib_npy.npy"
        np.save(calib_dataset, np.array(calib_images))
        print(f"   Created synthetic calibration dataset: {calib_dataset}")
    else:
        print(f"   Using {len(calib_files)} training images for calibration")
        # Load and preprocess images to 640x640
        import cv2
        calib_images = []
        for f in calib_files[:100]:  # Max 100 images
            img = cv2.imread(str(f))
            if img is not None:
                img = cv2.resize(img, (640, 640))
                calib_images.append(img)
        calib_dataset = "calib_npy.npy"
        np.save(calib_dataset, np.array(calib_images))
        print(f"   Saved {len(calib_images)} calibration images to {calib_dataset}")

    # Convert to RKNN
    print(f"\n[3] Converting to RKNN (int8 with calibration)...")
    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
    )

    print("   Loading split ONNX...")
    ret = rknn.load_onnx(model=str(ONNX_SPLIT))
    if ret != 0:
        print(f"   ERROR: load_onnx failed: {ret}")
        return False

    print("   Building RKNN model...")
    ret = rknn.build(
        do_quantization=True,
        dataset=calib_dataset,
    )
    if ret != 0:
        print(f"   ERROR: build failed: {ret}")
        return False

    print(f"   Exporting to {RKNN_OUT}...")
    ret = rknn.export_rknn(str(RKNN_OUT))
    if ret != 0:
        print(f"   ERROR: export failed: {ret}")
        return False

    rknn.release()
    print("   Done!")
    return True


def main():
    os.chdir(PROJECT_ROOT)

    # Step 1: Split ONNX
    if not split_onnx_output():
        return 1

    # Step 2: Convert to RKNN with calibration
    if not calibrate_and_convert():
        return 1

    print(f"\n{'='*60}")
    print(f"RKNN model with split outputs saved to: {RKNN_OUT}")
    print(f"Now update npu_inference.py and detector.py to handle dual outputs.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
