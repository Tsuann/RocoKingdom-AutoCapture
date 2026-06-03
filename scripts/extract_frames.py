#!/usr/bin/env python3
"""
视频帧提取工具 — 从游戏录像中提取精灵截图

用法:
    # 单视频提取 (每秒取一帧)
    python scripts/extract_frames.py video_xiao.mp4 --sprite xiao_dujiaoshou

    # 批量处理多个视频
    python scripts/extract_frames.py videos/ --sprite huzhu_quan

    # 控制提取密度
    python scripts/extract_frames.py video.mp4 --sprite yibei_er --fps 2 --max-frames 100

    # 去重模式 (跳过相似度过高的帧)
    python scripts/extract_frames.py video.mp4 --sprite emo_ding --dedup --threshold 0.95

输出:
    dataset/raw_{sprite_name}/video_0001.png
    dataset/raw_{sprite_name}/video_0002.png
    ...

建议的录像方式:
    - 在游戏中找到目标精灵
    - 围绕精灵旋转视角 (多角度)
    - 拉近拉远 (不同距离)
    - 等待精灵做一些动作 (不同姿态)
    - 每个精灵录 2-5 分钟即可
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# 精灵名称映射
SPRITE_KEYS = [
    "huzhu_quan",
    "yibei_er",
    "emo_ding",
    "juhua_li",
    "gongping_ge",
    "ling_hu",
    "xiao_dujiaoshou",
    "xiaoye_yifu",
]


def is_similar(img1: np.ndarray, img2: np.ndarray,
               threshold: float = 0.95) -> bool:
    """
    判断两张图是否过于相似 (用于去重)。
    使用感知哈希 (pHash) 快速比较。
    """
    # 缩放到 64x64 灰度图比较
    h, w = 64, 64
    a = cv2.resize(img1, (w, h))
    b = cv2.resize(img2, (w, h))
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # 计算结构相似度
    diff = np.mean(np.abs(a_gray.astype(float) - b_gray.astype(float))) / 255.0
    similarity = 1.0 - diff
    return similarity > threshold


def extract_frames(video_path: Path, output_dir: Path,
                   fps: float = 1.0,
                   max_frames: int = 200,
                   dedup: bool = True,
                   dedup_threshold: float = 0.95,
                   resize: tuple = None) -> int:
    """
    从视频中提取帧。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fps: 每秒提取帧数 (默认每1秒1帧)
        max_frames: 最大帧数上限
        dedup: 是否去重
        dedup_threshold: 去重相似度阈值
        resize: 可选缩放 (w, h)

    Returns: 提取的帧数
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"  视频: {video_path.name}")
    print(f"  时长: {duration:.1f}s, FPS: {video_fps:.1f}, "
          f"总帧数: {total_frames}")

    # 计算采样间隔
    if fps >= video_fps:
        # 如果要求的 fps 比视频高，每帧都取
        interval = 1
        print(f"  采样: 每帧 (视频 FPS={video_fps:.1f})")
    else:
        interval = max(1, int(video_fps / fps))
        print(f"  采样: 每 {interval} 帧 (~{fps} FPS)")

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_idx = 0
    prev_frame = None

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔采样
        if frame_idx % interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        # 可选缩放
        if resize:
            frame = cv2.resize(frame, resize)

        # 去重
        if dedup and prev_frame is not None:
            if is_similar(frame, prev_frame, dedup_threshold):
                continue

        prev_frame = frame.copy()

        # 保存
        filename = f"{video_path.stem}_{count:04d}.png"
        cv2.imwrite(str(output_dir / filename), frame)
        count += 1

        if count % 50 == 0:
            print(f"  已提取: {count} 帧")

    cap.release()
    print(f"  [OK] 提取 {count} 帧 -> {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="视频帧提取 — 从游戏录像中提取精灵截图"
    )
    parser.add_argument("input", help="视频文件或目录")
    parser.add_argument("--sprite", required=True,
                        help=f"精灵名称 ({'/'.join(SPRITE_KEYS)})")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="每秒提取帧数 (默认 2, 即每0.5秒一帧)")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="最大提取帧数 (默认 200)")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="去重相似帧 (默认开启)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="关闭去重")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="去重相似度阈值 (默认 0.95)")
    parser.add_argument("--output", default="dataset",
                        help="输出根目录 (默认 dataset)")
    parser.add_argument("--resize", default=None,
                        help="缩放尺寸 (如 1920x1080)")
    args = parser.parse_args()

    if args.sprite not in SPRITE_KEYS:
        print(f"[ERROR] 未知精灵: {args.sprite}")
        print(f"  可选: {SPRITE_KEYS}")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output) / f"raw_{args.sprite}"

    resize = None
    if args.resize:
        parts = args.resize.split("x")
        if len(parts) == 2:
            resize = (int(parts[0]), int(parts[1]))

    use_dedup = not args.no_dedup

    if input_path.is_file():
        # 单个视频
        print(f"\n[*] 提取: {input_path.name} -> {output_dir}")
        count = extract_frames(
            input_path, output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            dedup=use_dedup,
            dedup_threshold=args.threshold,
            resize=resize,
        )
    elif input_path.is_dir():
        # 批量处理目录
        videos = sorted(
            list(input_path.glob("*.mp4")) +
            list(input_path.glob("*.avi")) +
            list(input_path.glob("*.mov")) +
            list(input_path.glob("*.mkv")) +
            list(input_path.glob("*.webm"))
        )
        if not videos:
            print(f"[ERROR] 目录中没有视频文件: {input_path}")
            sys.exit(1)

        print(f"\n批量提取 {len(videos)} 个视频 -> {output_dir}")
        total = 0
        for v in videos:
            print(f"\n--- {v.name} ---")
            count = extract_frames(
                v, output_dir,
                fps=args.fps,
                max_frames=args.max_frames // len(videos) + 1,
                dedup=use_dedup,
                dedup_threshold=args.threshold,
                resize=resize,
            )
            total += count
        print(f"\n[DONE] 总计 {total} 帧 -> {output_dir}")
    else:
        print(f"[ERROR] 路径不存在: {input_path}")
        sys.exit(1)

    print(f"\n下一步:")
    print(f"  1. 浏览 {output_dir} 中的截图，删掉模糊/不相关的")
    print(f"  2. 标注: python scripts/label_tool.py {output_dir}")
    print(f"  3. 增量训练: python scripts/incremental_train.py "
          f"--new-data {output_dir} --device cuda")


if __name__ == "__main__":
    main()
