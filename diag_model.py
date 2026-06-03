#!/usr/bin/env python3
"""
诊断脚本：分析模型输出，确认检测不到框的原因。

用法: python diag_model.py [snapshot.png]
"""

import sys
import os
import numpy as np
import cv2

# 加载模型和测试图片
from npu_inference import NPUInference

def main():
    model_path = "models/sprite_detector.rknn"
    img_path = sys.argv[1] if len(sys.argv) > 1 else "snapshot_debug.png"

    print("=" * 60)
    print("模型输出诊断")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1] 加载模型: {model_path}")
    npu = NPUInference(model_path)
    print(f"    Input shape:  {npu.input_shape}")
    print(f"    Output shape: {npu.output_shape}")

    out_shape = npu.output_shape
    # 典型的 YOLO 输出: (1, C, A) 其中 C = 4 + num_classes (YOLOv8) 或 4 + 1 + num_classes (YOLOv5)
    num_channels = out_shape[1]
    num_anchors = out_shape[2]

    # 推测类别数
    # YOLOv8: C = 4 + N  => N = C - 4
    # YOLOv5: C = 5 + N  => N = C - 5
    n_v8 = num_channels - 4
    n_v5 = num_channels - 5
    print(f"    Output channels: {num_channels}")
    print(f"    如果是 YOLOv8 (无 objectness): {n_v8} 个类别")
    print(f"    如果是 YOLOv5 (有 objectness): {n_v5} 个类别")

    # 2. 加载测试图片
    print(f"\n[2] 加载测试图片: {img_path}")
    if not os.path.exists(img_path):
        print(f"    ⚠ 图片不存在! 使用黑色测试图")
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    else:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"    ⚠ 无法读取图片! 使用黑色测试图")
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            print(f"    Shape: {frame.shape}, mean={frame.mean():.1f}")

    # 3. 预处理 (与 RKNNDetector._preprocess_npu 一致)
    in_w, in_h = 640, 640
    f_h, f_w = frame.shape[:2]

    scale = min(in_w / f_w, in_h / f_h)
    new_w, new_h = int(f_w * scale), int(f_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    letterbox = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
    dx = (in_w - new_w) // 2
    dy = (in_h - new_h) // 2
    letterbox[dy:dy + new_h, dx:dx + new_w] = resized

    input_data = np.expand_dims(letterbox, axis=0)  # (1, 640, 640, 3) NHWC uint8
    print(f"    Preprocessed input: {input_data.shape}, dtype={input_data.dtype}")

    # 4. NPU 推理
    print(f"\n[3] NPU 推理...")
    output = npu.run(input_data)
    print(f"    Raw output shape: {output.shape}, dtype={output.dtype}")
    print(f"    Output min/max/mean: {output.min():.6f} / {output.max():.6f} / {output.mean():.6f}")

    # 5. 分析输出
    # output shape: (1, C, A) where C=5, A=8400
    o = output.squeeze(0)  # (C, A)
    print(f"\n[4] 输出分析 (squeezed shape: {o.shape}):")

    if o.shape[0] < o.shape[1]:
        o = o.T  # 转置为 (A, C)
        print(f"    转置后 shape: {o.shape}")

    # 按 YOLOv8 单类别解析：列 0-3 = bbox, 列 4 = class score
    if num_channels >= 5:
        bbox_raw = o[:, :4]
        class_scores = o[:, 4].copy()  # 单类别

        print(f"\n    Bbox (cx,cy,w,h) 统计 (前10个):")
        print(f"    {'idx':<6} {'cx':>8} {'cy':>8} {'w':>8} {'h':>8} {'score':>8}")
        print(f"    {'-'*50}")

        # 按 score 排序取前10
        top_idx = np.argsort(class_scores)[::-1][:10]
        for i in top_idx:
            cx, cy, w, h = bbox_raw[i]
            score = class_scores[i]
            marker = " ***" if score > 0.5 else ""
            print(f"    {i:<6} {cx:>8.4f} {cy:>8.4f} {w:>8.4f} {h:>8.4f} {score:>8.4f}{marker}")

        print(f"\n    类别分数统计:")
        print(f"      Max:       {class_scores.max():.6f}")
        print(f"      Min:       {class_scores.min():.6f}")
        print(f"      Mean:      {class_scores.mean():.6f}")
        print(f"      Median:    {np.median(class_scores):.6f}")
        print(f"      > 0.5:     {(class_scores > 0.5).sum()} / {len(class_scores)}")
        print(f"      > 0.3:     {(class_scores > 0.3).sum()} / {len(class_scores)}")
        print(f"      > 0.1:     {(class_scores > 0.1).sum()} / {len(class_scores)}")
        print(f"      > 0.01:    {(class_scores > 0.01).sum()} / {len(class_scores)}")

        print(f"\n    ╔══════════════════════════════════════════════════════╗")
        if class_scores.max() < 0.01:
            print(f"    ║  ⚠ 所有分数接近 0 — 模型可能完全没学到任何东西  ║")
        elif class_scores.max() < 0.5:
            print(f"    ║  ⚠ 最高分 {class_scores.max():.4f} < 0.5 — conf_threshold 太高了!  ║")
            print(f"    ║  建议: conf_threshold 降低到 {max(0.1, class_scores.max() * 0.7):.2f}           ║")
        else:
            over_05 = (class_scores > 0.5).sum()
            print(f"    ║  ✓ {over_05} 个 anchor 高于 0.5 阈值               ║")
            print(f"    ║  检查 NMS 后是否还有结果...                       ║")
        print(f"    ╚══════════════════════════════════════════════════════╝")

        # 6. 模拟完整后处理看最终结果
        print(f"\n[5] 模拟完整后处理 (conf_threshold=0.5):")
        from detector import yolo_postprocess

        for th in [0.5, 0.3, 0.1, 0.05, 0.01]:
            results = yolo_postprocess(
                output,
                input_shape=(640, 640),
                frame_shape=frame.shape,
                conf_threshold=th,
                nms_threshold=0.4,
                num_classes=1,  # 模型实际只有1类
            )
            print(f"    threshold={th:.2f}: {len(results)} detections")
            for r in results:
                print(f"      {r}")

    else:
        print(f"    意外的输出格式: {num_channels} 通道")

    # 7. 保存 letterbox 可视化，检查预处理
    cv2.imwrite("diag_letterbox.png", letterbox)
    print(f"\n[6] Letterbox 预处理结果保存到: diag_letterbox.png")

    npu.release()
    print("\n诊断完成.")


if __name__ == "__main__":
    main()
