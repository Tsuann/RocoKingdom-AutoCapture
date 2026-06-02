#!/usr/bin/env python3
"""
训练数据采集工具 — 从 HDMI 采集画面，按热键保存截图。

用法:
    python3 scripts/capture_screenshots.py [--output dataset/raw] [--device 0]

热键:
    SPACE  — 保存当前帧 (screenshot_0001.jpg, screenshot_0002.jpg, ...)
    ESC    — 退出

采集完成后，使用 label_tool.py 对截图进行标注。
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import cv2

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import HDMIStream
from utils import log


def main():
    parser = argparse.ArgumentParser(description="训练数据截图采集")
    parser.add_argument("--output", default="dataset/raw",
                        help="截图输出目录 (默认: dataset/raw)")
    parser.add_argument("--device", type=int, default=0,
                        help="HDMI 采集设备编号 (默认: 0)")
    parser.add_argument("--width", type=int, default=1920,
                        help="采集宽度 (默认: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="采集高度 (默认: 1080)")
    parser.add_argument("--fps", type=int, default=30,
                        help="采集帧率 (默认: 30)")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"截图将保存到: {output_dir.resolve()}")

    # 打开采集设备
    cap = HDMIStream(device=args.device, width=args.width,
                     height=args.height, fps=args.fps)
    cap.start()
    log.info("采集已启动，按 SPACE 保存截图，ESC 退出")

    count = 0
    overlay = "READY"

    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

            # 显示画面
            display = frame.copy()
            h, w = display.shape[:2]

            # 叠加信息
            cv2.putText(display, f"Saved: {count} | {overlay}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=Save  ESC=Exit",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            cv2.imshow("Screenshot Capture", display)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                # 保存原始帧（不含叠加信息）
                filename = output_dir / f"screenshot_{count:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                count += 1
                overlay = f"SAVED: {filename.name}"
                log.info(f"已保存: {filename.name} (总计 {count})")

            elif key == 27:  # ESC
                log.info("用户退出")
                break

            elif key == ord('r'):
                overlay = "READY"

    except KeyboardInterrupt:
        log.info("收到中断信号")
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        log.info(f"共保存 {count} 张截图到 {output_dir.resolve()}")


if __name__ == "__main__":
    main()
