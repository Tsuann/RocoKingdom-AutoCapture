"""
HDMI 画面采集模块

从 /dev/video0 (Rockchip HDMI RX) 采集游戏画面。
使用 GStreamer + V4L2 (io-mode=4 DMABUF) 管道抓帧，
通过后台 GLib 主循环持续采集，主线程通过 get_frame() 获取最新帧。

原因: Rockchip HDMI RX (rk_hdmirx) 驱动使用 multiplanar V4L2 API，
      OpenCV 的直接 read() 模式不兼容（会永久阻塞），必须使用
      GStreamer 的 io-mode=4 (DMABUF) 模式。
"""

import threading
import time
import sys
from collections import deque
from typing import Optional, Tuple

import numpy as np

from utils import log

# GStreamer — 避免与 Qt (OpenCV highgui) 的 GLib 主上下文冲突
import os as _os
_os.environ.setdefault("GST_GL_MAIN_CONTEXT", "0")

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# 初始化 GStreamer（全局一次性）
Gst.init(None)


class HDMIStream:
    """
    HDMI 输入流采集器（GStreamer 后端）。

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
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.queue_size = max(1, queue_size)

        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[Gst.Element] = None
        self._frame_queue: deque = deque(maxlen=queue_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._running = False
        self._frame_interval = 1.0 / fps if fps > 0 else 0

        # 统计
        self._frames_captured = 0
        self._last_frame_time = 0.0

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def start(self, display: bool = False) -> bool:
        """启动 GStreamer 采集管道。成功返回 True。

        display=True: 用 tee 分两路，一路 autovideosink/waylandsink 直接显示
                      （参考 test_hdmiin.sh），一路 appsink 供 ML 取帧。
        """
        if self._running:
            log.warning("Capture already running")
            return True

        device_path = f"/dev/video{self.device}"

        # 检测 display server，Wayland 上不能用 io-mode=4
        # （参考 test_hdmiin.sh：Wayland 分支去掉了 io-mode=4）
        ds = detect_display_server()
        io_mode = "" if ds == "wayland" else "io-mode=4"

        # 显示用的 sink（参考 test_hdmiin.sh）
        if ds == "wayland":
            display_sink = "waylandsink sync=false"
        else:
            display_sink = "autovideosink sync=false"

        if display:
            # 参考 test_hdmiin.sh：tee 分两路，每路各自 videoconvert
            #   t. → queue → videoconvert → NV12@原始 → videoscale → 1280x720 → autovideosink
            #   t. → queue → videoconvert → BGR → appsink (ML 取帧)
            pipeline_str = (
                f"v4l2src device={device_path} {io_mode} ! "
                f"tee name=t "
                f"t. ! queue ! videoconvert ! "
                f"video/x-raw,format=NV12,width={self.width},height={self.height} ! "
                f"videoscale ! video/x-raw,width=1280,height=720 ! "
                f"{display_sink} "
                f"t. ! queue ! videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name=sink emit-signals=true max-buffers=2 drop=true"
            )
        else:
            pipeline_str = (
                f"v4l2src device={device_path} {io_mode} ! "
                f"videoconvert ! "
                f"video/x-raw,format=BGR ! "
                f"appsink name=sink emit-signals=true max-buffers=2 drop=true"
            )

        log.info(f"GStreamer pipeline: {pipeline_str}")

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            log.error(f"Failed to create GStreamer pipeline: {e}")
            return False

        if self._pipeline is None:
            log.error("GStreamer pipeline is None")
            return False

        # 监听总线消息，方便排查管道错误
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # 获取 appsink 并配置回调
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            log.error("Failed to get appsink from pipeline")
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            return False

        self._appsink.set_property("emit-signals", True)
        self._appsink.connect("new-sample", self._on_new_sample)

        # 启动管道
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            log.error("Failed to start GStreamer pipeline")
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None
            return False

        self._running = True

        # 启动 GLib 主循环线程
        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._run_loop,
                                        name="HDMI-Gst-Loop",
                                        daemon=True)
        self._thread.start()

        # 等待首帧 (最多 5 秒)
        start_wait = time.monotonic()
        while time.monotonic() - start_wait < 5.0:
            with self._lock:
                if len(self._frame_queue) > 0:
                    break
            time.sleep(0.05)

        if len(self._frame_queue) > 0:
            log.info(f"HDMI capture started: {self.width}x{self.height}, "
                     f"actual FPS will be measured")
        else:
            log.warning("First frame not yet received after 5s")
            log.warning("Check HDMI connection — pipeline is running but no frames")

        return True

    def stop(self):
        """停止采集管道，释放资源。"""
        self._running = False

        # 停止 GLib 主循环
        if self._loop and self._loop.is_running():
            self._loop.quit()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # 停止管道
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None

        with self._lock:
            self._frame_queue.clear()

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
        """不拷贝直接返回引用（性能优先，调用方不应修改返回的帧）。"""
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

    def _on_bus_message(self, bus, message):
        """GStreamer 总线消息回调 — 输出错误和警告便于排查。"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error(f"GStreamer ERROR: {err.message}")
            log.debug(f"GStreamer debug: {debug}")
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            log.warning(f"GStreamer WARNING: {err.message}")
        elif t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            if new == Gst.State.PLAYING and old != Gst.State.PLAYING:
                log.info("GStreamer pipeline state → PLAYING")
        elif t == Gst.MessageType.STREAM_START:
            log.info("GStreamer stream started")

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """GStreamer appsink 回调 — 新帧到达。"""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        w = structure.get_value("width")
        h = structure.get_value("height")

        # 从 GstBuffer 提取 numpy 数组
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            # 构建 numpy 数组 (BGR 格式)
            frame = np.ndarray(
                shape=(h, w, 3),
                dtype=np.uint8,
                buffer=map_info.data,
            )
            # 必须拷贝！GStreamer 缓冲区在 unmap 后会被回收
            frame_copy = frame.copy()

            self._frames_captured += 1
            self._last_frame_time = time.perf_counter()

            with self._lock:
                self._frame_queue.append(frame_copy)

        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _run_loop(self):
        """后台 GLib 主循环。"""
        try:
            self._loop.run()
        except Exception as e:
            log.error(f"GLib main loop error: {e}")


# ============================================================
# HDMI 输入信息探测（参考 test_hdmiin.sh 的动态检测方式）
# ============================================================

def _run_cmd(args: list) -> str:
    """执行命令，返回 stdout。静默失败时返回空字符串。"""
    import subprocess
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL,
                                       timeout=5).decode("utf-8", errors="replace")
    except Exception:
        return ""


def detect_hdmi_device() -> int:
    """
    自动检测 HDMI RX 设备编号（参考 test_hdmiin.sh 的 v4l2-ctl 查询方式）。
    返回设备编号；找不到时返回 0（fallback）。
    """
    output = _run_cmd(["v4l2-ctl", "--list-devices"])
    if not output:
        log.warning("v4l2-ctl not available, fallback to /dev/video0")
        return 0

    # 匹配 "hdmirx" 段落中下一行的 /dev/videoN
    import re
    hdmirx_block = re.split(r'\n\n+', output)
    for block in hdmirx_block:
        if "hdmirx" in block.lower():
            m = re.search(r'/dev/video(\d+)', block)
            if m:
                dev_id = int(m.group(1))
                log.info(f"Auto-detected HDMI RX device: /dev/video{dev_id}")
                return dev_id

    log.warning("hdmirx device not found, fallback to /dev/video0")
    return 0


def detect_hdmi_resolution(device_id: int = None) -> Tuple[int, int]:
    """
    自动检测 HDMI 输入分辨率（参考 test_hdmiin.sh 的 --get-dv-timings）。
    返回 (width, height)。失败时返回 (1920, 1080) fallback。
    """
    if device_id is None:
        device_id = detect_hdmi_device()

    dev = f"/dev/video{device_id}"

    # 先触发一次 query 以刷新 DV timings
    _run_cmd(["v4l2-ctl", "-d", dev, "--set-dv-bt-timings", "query"])

    output = _run_cmd(["v4l2-ctl", "-d", dev, "--get-dv-timings"])
    if not output:
        log.warning("Cannot query DV timings, fallback to 1920x1080")
        return 1920, 1080

    import re
    w_match = re.search(r'Active width\s*:\s*(\d+)', output)
    h_match = re.search(r'Active height\s*:\s*(\d+)', output)  # 注意 test_hdmiin.sh 第15行拼写为 "heigh"

    if not w_match or not h_match:
        # 兼容驱动层可能的拼写
        h_match2 = re.search(r'Active heigh\w*\s*:\s*(\d+)', output)
        if h_match2:
            h_match = h_match2
        else:
            log.warning("Cannot parse DV timings, fallback to 1920x1080")
            return 1920, 1080

    w, h = int(w_match.group(1)), int(h_match.group(1))
    log.info(f"Auto-detected HDMI input resolution: {w}x{h}")
    return w, h


def detect_display_server() -> str:
    """
    检测当前 display server 类型（参考 test_hdmiin.sh 的 XDG_SESSION_TYPE）。
    返回 "wayland" 或 "x11"。
    """
    ds = _run_cmd(["bash", "-c", "echo $XDG_SESSION_TYPE"]).strip()
    if not ds:
        # Fallback：检查环境变量
        ds = _run_cmd(["loginctl", "show-session",
                       _run_cmd(["loginctl", "list-sessions", "--no-legend"])
                       .split()[0], "-p", "Type"]).strip()
        ds = ds.replace("Type=", "") if ds else ""

    if "wayland" in ds.lower():
        log.info("Display server: Wayland")
        return "wayland"
    log.info(f"Display server: {ds or 'x11 (assumed)'}")
    return "x11"


def detect_hdmi_audio_device() -> Optional[int]:
    """
    检测 HDMI RX 音频设备 card 号（参考 test_hdmiin.sh 的 arecord -l 查询）。
    返回 card 编号；未找到返回 None。
    """
    output = _run_cmd(["arecord", "-l"])
    if not output:
        return None

    import re
    for line in output.splitlines():
        if "hdmiin" in line.lower():
            m = re.search(r'card\s+(\d+)', line)
            if m:
                card = int(m.group(1))
                log.info(f"Auto-detected HDMI RX audio: card {card}")
                return card
    return None


# ============================================================
# 快捷测试函数
# ============================================================

def test_capture(show_display: bool = True, save_snapshot: bool = False):
    """
    测试 HDMI 采集是否正常。
    参考 test_hdmiin.sh 实现：
      - 自动检测设备/分辨率/display server
      - 用 GStreamer 原生 autovideosink/waylandsink 直接显示（不再依赖 OpenCV）
      - Ctrl+C 退出
    """
    import cv2
    import signal

    log.info("=" * 50)
    log.info("HDMI Capture Test (inspired by test_hdmiin.sh)")
    log.info("=" * 50)

    # 1. 自动检测 HDMI 设备 & 分辨率（参考 test_hdmiin.sh）
    device_id = detect_hdmi_device()
    width, height = detect_hdmi_resolution(device_id)
    display_server = detect_display_server()
    audio_card = detect_hdmi_audio_device()

    log.info(f"Device : /dev/video{device_id}")
    log.info(f"Input  : {width}x{height}")
    log.info(f"Display: {display_server}")
    log.info(f"Audio  : card {audio_card}" if audio_card is not None else "Audio  : N/A")

    # 2. 启动采集 + 原生显示（参考 test_hdmiin.sh 的 autovideosink）
    stream = HDMIStream(device=device_id, width=width, height=height, fps=60)
    if not stream.start(display=show_display):
        log.error("Test failed: cannot start capture")
        return False

    if show_display:
        log.info("GStreamer native display window opened (1280x720, like test_hdmiin.sh)")
    log.info("Press Ctrl+C to exit")

    # Ctrl+C 优雅退出
    shutdown = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown.set())

    try:
        frame_count = 0
        snapshot_saved = False
        fps_start = time.perf_counter()
        fps_frames = 0

        while stream.is_running and not shutdown.is_set():
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.002)
                continue

            frame_count += 1
            fps_frames += 1

            # 首帧截图
            if save_snapshot and not snapshot_saved:
                cv2.imwrite("test_snapshot.png", frame)
                log.info(f"Snapshot saved: test_snapshot.png ({frame.shape})")
                snapshot_saved = True

            # 每秒输出 FPS
            now = time.perf_counter()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps_val = fps_frames / elapsed
                stats = stream.stats
                log.info(f"FPS: {fps_val:.1f} | Frames: {stats['frames_captured']} | "
                         f"Queue: {stats['queue_size']}/{stream.queue_size}")
                fps_frames = 0
                fps_start = now

    finally:
        stream.stop()
        log.info(f"Test finished. Total frames captured: {frame_count}")

    return True


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging(level="INFO", log_dir="logs/")
    test_capture(display=True, save_snapshot=True)
