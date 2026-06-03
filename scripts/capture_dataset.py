#!/usr/bin/env python3
"""
训练数据采集工具 — Windows PC 版

从屏幕或摄像头捕获游戏截图，自动组织到 dataset/raw/ 目录。
支持热键快速截图，适合在固定点位大量采集精灵图像。

用法:
    python scripts/capture_dataset.py                    # 从摄像头采集
    python scripts/capture_dataset.py --screen           # 截取屏幕 (需要 mss)
    python scripts/capture_dataset.py --screen --region 0,0,1920,1080

快捷键:
    SPACE  — 保存当前画面
    C      — 连续截图模式 (每N帧自动保存一张)
    Q/ESC  — 退出

输出:
    dataset/raw/screenshot_0001.png
    dataset/raw/screenshot_0002.png
    ...
"""

import argparse
import sys
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_output_dir(base_dir: str = "dataset/raw") -> Path:
    """创建输出目录，以日期时间命名子文件夹。"""
    out = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = out / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


class ScreenCapture:
    """Windows 屏幕采集 (使用 mss 库)。"""

    def __init__(self, region=None):
        self.region = region  # (x, y, w, h) 或 None=全屏
        self._sct = None

    def start(self) -> bool:
        try:
            import mss
            self._sct = mss.mss()
            print(f"[INFO] 屏幕采集已就绪 (mss)")
            if self.region:
                print(f"  区域: {self.region}")
            return True
        except ImportError:
            print("[ERROR] 需要安装 mss: pip install mss")
            return False
        except Exception as e:
            print(f"[ERROR] 屏幕采集初始化失败: {e}")
            return False

    def get_frame(self):
        """捕获一帧屏幕画面 (BGR numpy array)。"""
        if self._sct is None:
            return None
        try:
            if self.region:
                x, y, w, h = self.region
                monitor = {"top": y, "left": x, "width": w, "height": h}
            else:
                monitor = self._sct.monitors[1]  # 主显示器

            img = self._sct.grab(monitor)
            import numpy as np
            frame = np.array(img)
            # BGRA → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"[WARN] 截图失败: {e}")
            return None

    def stop(self):
        self._sct = None


class WebcamCapture:
    """摄像头采集。"""

    def __init__(self, device=0, width=1920, height=1080):
        self.device = device
        self.width = width
        self.height = height
        self._cap = None

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {self.device}")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[INFO] 摄像头已就绪: {actual_w:.0f}x{actual_h:.0f}")
        return True

    def get_frame(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def stop(self):
        if self._cap:
            self._cap.release()
            self._cap = None


def main():
    parser = argparse.ArgumentParser(
        description="训练数据采集工具 — 截取游戏画面用于标注训练"
    )
    parser.add_argument("--screen", action="store_true",
                        help="使用屏幕截图 (默认使用摄像头)")
    parser.add_argument("--region", default=None,
                        help="屏幕区域 x,y,w,h (如: 0,0,1920,1080)")
    parser.add_argument("--device", type=int, default=0,
                        help="摄像头设备编号 (默认 0)")
    parser.add_argument("--output", default="dataset/raw",
                        help="输出目录 (默认 dataset/raw)")
    parser.add_argument("--width", type=int, default=1920,
                        help="采集宽度 (默认 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="采集高度 (默认 1080)")
    parser.add_argument("--continuous-interval", type=float, default=2.0,
                        help="连续截图模式间隔秒数 (默认 2.0)")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = create_output_dir(args.output)
    print(f"[INFO] 截图保存至: {output_dir}")

    # 初始化采集
    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) == 4:
            region = tuple(parts)
        else:
            print("[ERROR] --region 格式: x,y,w,h")
            sys.exit(1)

    if args.screen:
        cap = ScreenCapture(region=region)
    else:
        cap = WebcamCapture(
            device=args.device,
            width=args.width,
            height=args.height,
        )

    if not cap.start():
        sys.exit(1)

    print("=" * 60)
    print("快捷键:")
    print("  SPACE  — 保存当前截图")
    print("  C      — 切换连续截图模式")
    print("  1-8    — 标记当前截图属于哪个精灵")
    print("  Q/ESC  — 退出")
    print("=" * 60)

    # 精灵快捷键对应
    sprite_hotkeys = {
        ord('1'): "huzhu_quan",
        ord('2'): "yibei_er",
        ord('3'): "emo_ding",
        ord('4'): "juhua_li",
        ord('5'): "gongping_ge",
        ord('6'): "ling_hu",
        ord('7'): "xiao_dujiaoshou",
        ord('8'): "xiaoye_yifu",
    }
    current_sprite = None  # 当前标记的精灵

    cv2.namedWindow("Dataset Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Capture", 960, 540)

    screenshot_count = 0
    continuous_mode = False
    last_capture_time = 0

    # 状态栏
    sprite_names = [
        "1:护主犬", "2:伊贝儿", "3:恶魔叮", "4:菊花梨",
        "5:公平鸽", "6:灵狐", "7:小独角兽", "8:小夜/朔夜伊芙"
    ]

    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 显示信息叠加
            display = frame.copy()
            h, w = display.shape[:2]

            # 半透明状态栏
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # 状态信息
            status = f"Screenshots: {screenshot_count} | "
            status += f"Continuous: {'ON' if continuous_mode else 'OFF'} | "
            if current_sprite:
                status += f"Marking as: {current_sprite}"
            else:
                status += "No sprite label"
            cv2.putText(display, status, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # 精灵快捷键提示
            for i, name in enumerate(sprite_names):
                y = 25 + i * 20
                color = (0, 255, 0) if current_sprite and i == list(
                    sprite_hotkeys.values()).index(current_sprite) else (180, 180, 180)
                cv2.putText(display, name, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.imshow("Dataset Capture", display)

            # 连续截图
            if continuous_mode and time.time() - last_capture_time > args.continuous_interval:
                last_capture_time = time.time()
                # 自动保存
                prefix = current_sprite or "unknown"
                filename = f"{prefix}_{screenshot_count:04d}.png"
                save_path = output_dir / filename
                cv2.imwrite(str(save_path), frame)
                screenshot_count += 1
                print(f"  [{screenshot_count}] 自动保存: {filename}")

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC / Q
                print("退出采集")
                break

            elif key == 32:  # SPACE — 手动截图
                prefix = current_sprite or "unknown"
                filename = f"{prefix}_{screenshot_count:04d}.png"
                save_path = output_dir / filename
                cv2.imwrite(str(save_path), frame)
                screenshot_count += 1
                print(f"  [{screenshot_count}] 已保存: {filename}")

            elif key == ord('c'):  # 切换连续模式
                continuous_mode = not continuous_mode
                last_capture_time = time.time()
                print(f"  连续截图: {'ON' if continuous_mode else 'OFF'}")

            elif key in sprite_hotkeys:  # 精灵标记
                current_sprite = sprite_hotkeys[key]
                print(f"  当前标记精灵: {current_sprite}")

    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print(f"\n采集完成！共保存 {screenshot_count} 张截图")
        print(f"输出目录: {output_dir.resolve()}")
        print(f"\n下一步:")
        print(f"  python scripts/label_tool.py {output_dir} "
              f"--classes huzhu_quan,yibei_er,emo_ding,juhua_li,"
              f"gongping_ge,ling_hu,xiao_dujiaoshou,xiaoye_yifu")


if __name__ == "__main__":
    main()
