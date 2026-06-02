"""
工具函数模块 — 日志、坐标变换、帧绘制等通用功能。
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


# ============================================================
# 日志
# ============================================================

def setup_logging(level: str = "INFO", log_dir: str = "logs/") -> logging.Logger:
    """配置日志系统，同时输出到控制台和文件。"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("roco_auto")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 文件 handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"run_{timestamp}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = logging.getLogger("roco_auto")


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# 坐标与变换
# ============================================================

def screen_to_game_coord(screen_x: int, screen_y: int,
                          capture_width: int = 1920,
                          capture_height: int = 1080,
                          game_width: int = 1920,
                          game_height: int = 1080) -> Tuple[int, int]:
    """将采集画面坐标映射到游戏坐标 (如需缩放)。"""
    scale_x = game_width / capture_width
    scale_y = game_height / capture_height
    return int(screen_x * scale_x), int(screen_y * scale_y)


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """计算边界框中心点。bbox 格式: (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """计算边界框面积。"""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """计算两点欧氏距离。"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def clip_to_frame(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    """将坐标裁剪到画面范围内。"""
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


# ============================================================
# 帧绘制 (debug 用)
# ============================================================

def draw_detections(frame: np.ndarray,
                    detections: list,
                    target_idx: Optional[int] = None,
                    tracker_info: Optional[dict] = None) -> np.ndarray:
    """在帧上绘制检测框和追踪信息。"""
    vis = frame.copy()
    colors = {
        "normal": (0, 255, 0),       # 绿色
        "shiny": (0, 255, 255),      # 黄色
        "corrupted": (0, 0, 255),    # 红色
    }

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
        class_name = det.get("class", "unknown")
        conf = det.get("confidence", 0.0)
        color = colors.get(class_name, (255, 255, 255))

        thickness = 3 if i == target_idx else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # 标签
        label = f"{class_name} {conf:.2f}"
        if i == target_idx:
            label = ">>> " + label

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 中心点
        if i == target_idx:
            cx, cy = bbox_center((x1, y1, x2, y2))
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)

    # 追踪信息
    if tracker_info:
        pred = tracker_info.get("predicted_position")
        if pred:
            cv2.circle(vis, (int(pred[0]), int(pred[1])), 6, (255, 0, 255), -1)

    # 状态文字
    status = tracker_info.get("status", "") if tracker_info else ""
    if status:
        cv2.putText(vis, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # FPS
    fps = tracker_info.get("fps", 0) if tracker_info else 0
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return vis


# ============================================================
# 计时器
# ============================================================

class FPSCounter:
    """FPS 计数器。"""

    def __init__(self, window_size: int = 30):
        self._times = []
        self._window = window_size

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


class Timer:
    """简易计时器，用于状态超时判断。"""

    def __init__(self):
        self._start = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def expired(self, timeout: float) -> bool:
        return self.elapsed > timeout


if __name__ == "__main__":
    # 简单自测
    cfg = load_config("config.yaml")
    print(f"Loaded config: {list(cfg.keys())}")
    log.info("Utils module loaded OK")
