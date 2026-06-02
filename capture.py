"""
HDMI 画面采集模块

从 /dev/video0 (Rockchip HDMI RX) 采集游戏画面。
使用后台线程持续抓帧，主线程通过 get_frame() 获取最新帧。
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from utils import log


class HDMIStream:
    """
    HDMI 输入流采集器。

    用法:
        stream = HDMIStream(width=1920, height=1080, fps=30)
        stream.start()
        frame = stream.get_frame()
        stream.stop()
    """

    def __init__(self,
                 device: int = 0,
                 width: int = 1920,
                 height: int = 1080,
                 fps: int = 30,
                 queue_size: int = 2):
        """
        Args:
            device: /dev/video 编号 (0 = /dev/video0)
            width, height: 期望分辨率
            fps: 目标帧率
            queue_size: 帧队列最大长度 (1 = 只保留最新帧)
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.queue_size = max(1, queue_size)

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: deque = deque(maxlen=queue_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_interval = 1.0 / fps if fps > 0 else 0

        # 统计
        self._frames_captured = 0
        self._last_frame_time = 0.0

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def start(self) -> bool:
        """启动采集线程。成功返回 True。"""
        if self._running:
            log.warning("Capture already running")
            return True

        # 1. 打开设备
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            log.error(f"Failed to open /dev/video{self.device}")
            self._cap = None
            return False

        # 2. 设置参数
        # RK3588 HDMI RX 默认输出 BGR3 (BGR 8-8-8)
        # 但 OpenCV 可能需要明确设置 FOURCC
        fourcc = cv2.VideoWriter_fourcc(*"BGR3")
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 3. 测试读取一帧
        ret, frame = self._cap.read()
        if not ret or frame is None:
            log.error("Failed to read first frame from HDMI input")
            self._cap.release()
            self._cap = None
            return False

        log.info(f"HDMI capture started: {frame.shape[1]}x{frame.shape[0]}, "
                 f"actual FPS will be measured")

        # 4. 启动采集线程
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        name="HDMI-Capture",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """停止采集线程，释放资源。"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._lock:
            self._frame_queue.clear()

        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

        log.info(f"HDMI capture stopped. Total frames: {self._frames_captured}")

    # ----------------------------------------------------------
    # 帧获取
    # ----------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        """返回最新采集帧 (BGR ndarray)。无可用帧时返回 None。"""
        with self._lock:
            if len(self._frame_queue) > 0:
                return self._frame_queue[-1].copy()
            return None

    def get_frame_nonblock(self) -> Optional[np.ndarray]:
        """不拷贝直接返回引用 (性能优先，但不要修改返回的帧)。"""
        with self._lock:
            if len(self._frame_queue) > 0:
                return self._frame_queue[-1]
            return None

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        return {
            "frames_captured": self._frames_captured,
            "queue_size": len(self._frame_queue),
        }

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _capture_loop(self):
        """后台采集循环。"""
        last_log = 0.0

        while self._running:
            loop_start = time.perf_counter()

            if self._cap is None or not self._cap.isOpened():
                log.error("Capture device lost, stopping...")
                self._running = False
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                # 偶发的空帧，跳过
                time.sleep(0.001)
                continue

            self._frames_captured += 1
            self._last_frame_time = loop_start

            with self._lock:
                self._frame_queue.append(frame)

            # 每秒输出一次统计
            if loop_start - last_log > 10.0:
                last_log = loop_start
                log.debug(f"Capture stats: {self.stats}")

            # 帧率限制
            elapsed = time.perf_counter() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ============================================================
# 快捷测试函数
# ============================================================

def test_capture(display: bool = True, save_snapshot: bool = False):
    """
    测试 HDMI 采集是否正常。
    按 ESC 退出显示窗口。
    """
    log.info("Testing HDMI capture...")
    stream = HDMIStream(device=0, width=1920, height=1080, fps=30)

    if not stream.start():
        log.error("Test failed: cannot start capture")
        return False

    log.info("Capture started. Press ESC in the display window to exit.")

    try:
        while stream.is_running:
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if save_snapshot:
                cv2.imwrite("test_snapshot.png", frame)
                log.info("Snapshot saved to test_snapshot.png")
                save_snapshot = False  # 只存一张

            if display:
                # 缩小显示以适应屏幕
                disp = cv2.resize(frame, (960, 540))
                cv2.putText(disp, f"Frames: {stream.stats['frames_captured']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("HDMI Capture Test", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
    finally:
        stream.stop()
        if display:
            cv2.destroyAllWindows()

    log.info("Test completed successfully")
    return True


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging(level="INFO", log_dir="logs/")
    test_capture(display=True, save_snapshot=True)
