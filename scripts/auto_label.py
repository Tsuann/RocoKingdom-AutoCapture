#!/usr/bin/env python3
"""
自动标注工具 — 利用运动检测自动生成初始标注框

适用于: 玩家围绕精灵旋转录制的视频提取的帧
原理:   精灵是画面中主要移动目标，通过帧差法定位

用法:
    python scripts/auto_label.py dataset/raw_xiao_dujiaoshou/ --class-id 6
    # class-id: 0=护主犬 1=伊贝儿 2=恶魔叮 3=菊花梨
    #           4=公平鸽 5=灵狐 6=小独角兽 7=小夜

输出:
    dataset/raw_xiao_dujiaoshou/labels/  (YOLO格式 .txt)
    dataset/labeled_xiao_dujiaoshou/     (整理后的 images + labels)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPRITE_NAMES = [
    "huzhu_quan", "yibei_er", "emo_ding", "juhua_li",
    "gongping_ge", "ling_hu", "xiao_dujiaoshou", "xiaoye_yifu",
]


def find_moving_object(frame: np.ndarray,
                       prev_frame: np.ndarray,
                       min_area: int = 2000) -> tuple:
    """
    通过帧差法找到运动目标。

    Returns: (x, y, w, h) 或 None
    """
    # 灰度化 + 模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    # 帧差
    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 膨胀连接相邻区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 取面积最大的轮廓
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


def find_salient_region(frame: np.ndarray,
                         min_area: int = 3000) -> tuple:
    """
    备用方案: 基于边缘检测 + 轮廓找到画面中最突出的区域。
    适用于帧差法失败时。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 取面积最大的轮廓区域
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # 排除画面边缘的大轮廓
    for cnt in contours[:5]:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        if area > frame_area * 0.5:  # 超过一半画面，忽略
            continue
        if area >= min_area:
            return (x, y, w, h)

    return None


def auto_label_frames(frames_dir: Path, class_id: int,
                       output_dir: Path = None) -> int:
    """
    对目录中的所有帧进行自动标注。

    Args:
        frames_dir: 帧图片目录
        class_id: 精灵类别 ID (0-7)
        output_dir: 输出目录 (默认 frames_dir 同级 labeled_ 目录)

    Returns: 标注数量
    """
    images = sorted(
        list(frames_dir.glob("*.png")) +
        list(frames_dir.glob("*.jpg"))
    )
    if len(images) < 2:
        print(f"[ERROR] 至少需要2张图片，当前: {len(images)}")
        return 0

    if output_dir is None:
        output_dir = frames_dir.parent / f"labeled_{frames_dir.name.replace('raw_', '')}"

    img_out = output_dir / "images"
    lbl_out = output_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"自动标注: {len(images)} 张图片 -> {output_dir}")
    print(f"  精灵类别: [{class_id}] {SPRITE_NAMES[class_id]}")

    labeled = 0
    prev_frame = None

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        bbox = None

        # 方法1: 帧差法 (与前一帧比较)
        if prev_frame is not None:
            bbox = find_moving_object(frame, prev_frame, min_area=2000)

        # 方法2: 显著性检测 (备用)
        if bbox is None:
            bbox = find_salient_region(frame, min_area=3000)

        prev_frame = frame.copy()

        if bbox is None:
            continue  # 找不到目标，跳过

        x, y, bw, bh = bbox
        # YOLO 格式: class_id cx cy w h (归一化)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        # 过滤不合理的框 (>60% 画面 或 <2% 画面)
        if nw > 0.6 or nh > 0.6 or nw < 0.02 or nh < 0.02:
            continue

        # 保存图片
        out_name = f"{img_path.stem}"
        cv2.imwrite(str(img_out / f"{out_name}.png"), frame)

        # 保存标注
        with open(lbl_out / f"{out_name}.txt", "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        labeled += 1

        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(images)}")

    print(f"\n[DONE] 自动标注完成: {labeled}/{len(images)} 张")
    print(f"  输出: {output_dir}")
    print(f"\n  -> 请运行 label_tool 检查修正自动标注结果:")
    print(f"     python scripts/label_tool.py {output_dir} --classes {','.join(SPRITE_NAMES)}")
    return labeled


def main():
    parser = argparse.ArgumentParser(
        description="自动标注 — 运动检测 + 显著性检测生成初始标注"
    )
    parser.add_argument("frames_dir", help="帧图片目录")
    parser.add_argument("--class-id", type=int, required=True,
                        help=f"精灵类别ID: 0={SPRITE_NAMES[0]} ... 7={SPRITE_NAMES[7]}")
    parser.add_argument("--output", default=None,
                        help="输出目录 (默认自动)")
    args = parser.parse_args()

    if args.class_id < 0 or args.class_id > 7:
        print(f"[ERROR] class-id 范围 0-7: {list(enumerate(SPRITE_NAMES))}")
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"[ERROR] 目录不存在: {frames_dir}")
        sys.exit(1)

    auto_label_frames(frames_dir, args.class_id, args.output)


if __name__ == "__main__":
    main()
