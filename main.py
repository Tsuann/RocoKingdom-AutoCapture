#!/usr/bin/env python3
"""
洛克王国：世界 — 自动捕捉精灵系统

入口脚本。支持三种运行模式:
  python main.py debug    — 调试模式, 显示画面+检测框
  python main.py manual   — 手动模式, 只检测不操作
  python main.py auto     — 全自动模式

依赖:
  - HDMI 输入 (/dev/video0) 已连接 Windows PC
  - (auto 模式) USB HID gadget 已配置: sudo bash setup_gadget.sh rockchip-mouse
"""

import argparse
import os as _os
import signal
import sys
import threading
import time
from pathlib import Path

# 抑制 OpenCV Qt 后端和 GStreamer GLib 的线程警告
_os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")
_os.environ.setdefault("QT_FATAL_WARNINGS", "0")
_os.environ.setdefault("G_MESSAGES_DEBUG", "")
# 强制 Qt 使用软件光栅渲染，避免 X11 QPixmap 内存泄漏
_os.environ.setdefault("QT_QUICK_BACKEND", "software")
_os.environ.setdefault("QT_OPENGL", "software")

import cv2

from utils import (
    log, setup_logging, load_config,
    FPSCounter, draw_detections,
)


class RocoAutoCapture:
    """主应用程序。"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.stop_event = threading.Event()
        self.fps = FPSCounter()

        # 模块延迟初始化
        self._stream = None
        self._detector = None
        self._tracker = None
        self._controller = None
        self._pipeline = None
        self._hotkey_listener = None

    # ----------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------

    def _init_capture(self, display: bool = False):
        from capture import HDMIStream
        cap_cfg = self.config.get("capture", {})
        self._stream = HDMIStream(
            device=cap_cfg.get("device", 0),
            width=cap_cfg.get("width", 1920),
            height=cap_cfg.get("height", 1080),
            fps=cap_cfg.get("fps", 30),
        )
        if not self._stream.start(display=display):
            log.error("Failed to start HDMI capture")
            return False
        return True

    def _init_detector(self):
        from detector import SpriteDetector
        self._detector = SpriteDetector(self.config.get("detector", {}))
        log.info(f"Detector backend: {self._detector.backend}")
        return True

    def _init_controller(self):
        from controller import GameController
        self._controller = GameController(self.config)

        if not self._controller.open():
            log.warning("Controller not available. "
                        "USB gadget may not be configured.")
            self._controller = None
            return False
        return True

    def _init_pipeline(self):
        from state_machine import CapturePipeline, HotkeyListener, Event
        self._pipeline = CapturePipeline(
            self._detector, self._controller, self.config
        )
        self._pipeline.set_frame_provider(self._stream.get_frame)

        # 热键监听
        hotkey_cfg = self.config.get("hotkeys", {})
        hotkey_map = {
            hotkey_cfg.get("start", "f2"): Event.START,
            hotkey_cfg.get("pause", "f3"): Event.PAUSE,
            hotkey_cfg.get("exit", "f4"): Event.STOP,
        }

        self._hotkey_listener = HotkeyListener(hotkey_map)
        self._hotkey_listener.start()

    # ----------------------------------------------------------
    # 模式: DEBUG
    # ----------------------------------------------------------

    def run_debug(self):
        """调试模式: 显示采集画面, 叠加检测框和追踪信息。"""
        log.info("=" * 60)
        log.info("DEBUG MODE — display capture + detections")
        log.info("  ESC  : exit")
        log.info("  SPACE: save snapshot")
        log.info("  T    : toggle template matching on/off")
        log.info("=" * 60)

        # 诊断：检查显示环境
        import os as _os
        _display = _os.environ.get("DISPLAY", "")
        log.info(f"DISPLAY={_display or '(not set!)'}")

        if not _display:
            log.error("No DISPLAY environment variable set!")
            log.error("If using SSH, reconnect with: ssh -X orangepi@host")
            log.error("Or run directly on the Orange Pi's terminal with a monitor attached.")
            return

        # 必须在 GStreamer 之前创建 OpenCV 窗口，否则 Qt/GLib 主上下文冲突
        log.info("Creating debug window (960x540) before GStreamer init...")
        cv2.namedWindow("RocoAutoCapture — DEBUG", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RocoAutoCapture — DEBUG", 960, 540)

        if not self._init_capture():
            return

        # 等待第一帧确保采集正常
        log.info("Waiting for first frame...")
        for i in range(50):
            frame = self._stream.get_frame()
            if frame is not None:
                log.info(f"First frame OK: shape={frame.shape}, mean={frame.mean():.1f}")
                break
            time.sleep(0.05)
        else:
            log.error("No frame received after 2.5s! Check HDMI connection.")
            self._stream.stop()
            return

        self._init_detector()

        from tracker import MultiTracker
        trk_cfg = self.config.get("tracker", {})
        tracker = MultiTracker(
            max_disappeared=trk_cfg.get("max_disappeared", 30),
            process_noise=trk_cfg.get("kalman", {}).get("process_noise", 0.03),
            measurement_noise=trk_cfg.get("kalman", {}).get("measurement_noise", 0.1),
        )

        show_templates = True

        log.info("GStreamer capture started. Press ESC to exit, SPACE to save snapshot.")

        try:
            while not self.stop_event.is_set():
                frame = self._stream.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                self.fps.tick()

                # 检测精灵
                sprites = self._detector.detect_sprites(frame)
                sprites = tracker.update(sprites)

                # 检测 UI (可选)
                ui_results = []
                if show_templates:
                    ui_results = self._detector.detect_ui(frame)

                # 所有检测结果
                all_dets = sprites + [{
                    "bbox": r["bbox"],
                    "class": "ui",
                    "confidence": r["confidence"],
                    "source": "template",
                    "name": r.get("name", ""),
                } for r in ui_results]

                # 选择目标
                target = tracker.select_target(
                    prefer=["shiny", "corrupted", "normal"],
                    detections=sprites,
                )
                target_idx = None
                if target and sprites:
                    target_id = target.get("tracker_id")
                    for i, s in enumerate(sprites):
                        if s.get("tracker_id") == target_id:
                            target_idx = i
                            break

                # 绘制
                from utils import draw_detections
                vis = draw_detections(
                    frame, all_dets, target_idx=target_idx,
                    tracker_info={
                        "status": f"Backend: {self._detector.backend} | "
                                  f"Sprites: {len(sprites)} | "
                                  f"Trackers: {tracker.active_count} | "
                                  f"Inference: {self._detector.inference_time * 1000:.0f}ms",
                        "fps": self.fps.fps,
                    }
                )

                # 缩放到 960x540 显示
                disp = cv2.resize(vis, (960, 540))
                cv2.imshow("RocoAutoCapture — DEBUG", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    cv2.imwrite("snapshot_debug.png", vis)
                    log.info("Snapshot saved: snapshot_debug.png")
                elif key == ord('t'):
                    show_templates = not show_templates
                    log.info(f"Template display: {'ON' if show_templates else 'OFF'}")

        finally:
            self._stream.stop()
            cv2.destroyAllWindows()
            self._detector.release()

    # ----------------------------------------------------------
    # 模式: DEBUG-AUTO (debug 显示 + auto 控制)
    # ----------------------------------------------------------

    def run_debug_auto(self):
        """Debug-Auto 模式: 显示循环只负责渲染画面，FSM 线程负责检测+控制。
        两个线程不竞争 NPU，不互相阻塞。
        """
        log.info("=" * 60)
        log.info("DEBUG-AUTO MODE — debug display + auto control")
        log.info("  ESC  : exit")
        log.info("  F3   : pause/resume")
        log.info("=" * 60)

        import os as _os
        if not _os.environ.get("DISPLAY"):
            log.error("No DISPLAY!"); return

        cv2.namedWindow("RocoAutoCapture — DEBUG-AUTO", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RocoAutoCapture — DEBUG-AUTO", 960, 540)

        if not self._init_capture(): return

        for i in range(50):
            frame = self._stream.get_frame()
            if frame is not None: break
            time.sleep(0.05)
        else:
            log.error("No frame!"); self._stream.stop(); return

        if not self._stream.wait_for_fresh_frames(min_frames=3, timeout=5.0):
            log.error("Capture produced only the first frame; HDMI/GStreamer stream is not advancing.")
            self._stream.stop()
            return

        self._init_detector()
        self._init_controller()

        from state_machine import CapturePipeline, HotkeyListener, Event
        self._pipeline = CapturePipeline(self._detector, self._controller, self.config)
        self._pipeline.set_frame_provider(self._stream.get_frame)

        tracker = self._pipeline.tracker  # 共享 tracker

        # 热键
        hotkey_cfg = self.config.get("hotkeys", {})
        self._hotkey_listener = HotkeyListener({
            hotkey_cfg.get("pause", "f3"): Event.PAUSE,
            hotkey_cfg.get("exit", "f4"): Event.STOP,
        })
        self._hotkey_listener.start()

        # 启动 FSM 线程
        fsm_stop = threading.Event()
        fsm_thread = threading.Thread(
            target=lambda: self._pipeline.run(fsm_stop, show_display=False),
            name="FSM", daemon=True)
        fsm_thread.start()

        # 热键处理
        def _hotkeys():
            while not self.stop_event.is_set():
                evt = self._hotkey_listener.get_event() if self._hotkey_listener else None
                if evt:
                    if evt.value == "stop":
                        self.stop_event.set(); fsm_stop.set()
                    elif evt.value == "pause":
                        if self._pipeline.fsm.is_paused:
                            self._pipeline.fsm.resume()
                        else:
                            self._pipeline.fsm.pause()
                time.sleep(0.05)
        threading.Thread(target=_hotkeys, daemon=True).start()

        # 主循环：只负责显示，不做检测（检测由 FSM 线程完成）
        # 使用 get_frame_nonblock() 避免拷贝（draw_detections 内部会 copy）
        log.info("Display loop starting (FSM handles detection)...")
        _last_health_check = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                frame = self._stream.get_frame_nonblock()
                if frame is None:
                    time.sleep(0.002); continue

                self.fps.tick()

                # 管道健康检查：每 2 秒检测一次是否卡死
                now = time.perf_counter()
                if now - _last_health_check > 2.0:
                    _last_health_check = now
                    if self._stream.is_stale:
                        log.error(
                            f"⚠️ PIPELINE STALLED! "
                            f"frames_captured={self._stream.stats['frames_captured']}, "
                            f"last_frame={now - self._stream._last_frame_time:.1f}s ago")
                        # 尝试自动重启管道
                        log.info("Attempting pipeline restart...")
                        try:
                            if self._stream.restart(display=False):
                                log.info("Pipeline restarted successfully")
                                # 重置 last_frame_time 引用（restart 会重建对象属性）
                                time.sleep(0.3)
                                continue
                            else:
                                log.error("Pipeline restart FAILED")
                                self.stop_event.set()
                                fsm_stop.set()
                                break
                        except Exception as re:
                            log.error(f"Pipeline restart error: {re}")
                            self.stop_event.set()
                            fsm_stop.set()
                            break

                # 从 FSM 读取最新状态（不调用 NPU，不加锁）
                state_name = self._pipeline.fsm.state.value
                target = self._pipeline._current_target
                throws = f"{self._pipeline._throw_count}/{self._pipeline.max_throw_attempts}"

                # 线程安全地读取活跃追踪器
                display_dets = tracker.get_active_detections()

                # 选目标
                target_idx = None
                if target:
                    tid = target.get("tracker_id")
                    if tid is not None:
                        for i, d in enumerate(display_dets):
                            if d.get("tracker_id") == tid:
                                target_idx = i; break

                # 从 pipeline 读取最新检测结果用于框显示
                # （FSM _on_scan 等会更新 tracker，
                #   这里只读取不修改，无竞争）
                vis = draw_detections(frame, display_dets, target_idx=target_idx,
                    tracker_info={
                        "status": (f"State: {state_name} | "
                                   f"Backend: {self._detector.backend} | "
                                   f"Inf: {self._detector.inference_time*1000:.0f}ms | "
                                   f"Throws: {throws}"),
                        "fps": self.fps.fps,
                    })
                disp = cv2.resize(vis, (960, 540))
                cv2.imshow("RocoAutoCapture — DEBUG-AUTO", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    cv2.imwrite("snapshot_debug_auto.png", vis)
                    log.info("Snapshot saved")
        finally:
            fsm_stop.set()
            self._stream.stop()
            self._detector.release()
            if self._controller: self._controller.close()
            if self._hotkey_listener: self._hotkey_listener.stop()
            cv2.destroyAllWindows()

        if self._pipeline:
            log.info(f"Stats: {self._pipeline.stats}")

    # ----------------------------------------------------------
    # 模式: MANUAL
    # ----------------------------------------------------------

    def run_manual(self):
        """手动模式: 检测并显示, 但不自动操作。"""
        log.info("=" * 60)
        log.info("MANUAL MODE — detection only, no automatic actions")
        log.info("  Detection results printed to console")
        log.info("  Ctrl+C to exit")
        log.info("=" * 60)

        if not self._init_capture():
            return

        self._init_detector()

        try:
            while not self.stop_event.is_set():
                frame = self._stream.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                self.fps.tick()

                # 检测
                sprites = self._detector.detect_sprites(frame)
                ui_elements = self._detector.detect_ui(frame)

                # 输出结果
                for s in sprites:
                    cls = s.get("class", "?")
                    conf = s.get("confidence", 0)
                    bbox = s.get("bbox", (0, 0, 0, 0))
                    log.info(f"[{cls}] conf={conf:.2f} bbox={bbox}")

                for ui in ui_elements:
                    log.info(f"[UI:{ui.get('name', '?')}] "
                             f"conf={ui.get('confidence', 0):.2f}")

                if not sprites and not ui_elements:
                    log.debug(f"No detections | FPS:{self.fps.fps:.1f}")

                time.sleep(0.5)

        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            self._stream.stop()
            self._detector.release()

    # ----------------------------------------------------------
    # 模式: AUTO
    # ----------------------------------------------------------

    def run_auto(self, show_display: bool = False):
        """全自动模式: 检测 + 追踪 + 控制 + 实时画面显示。

        Args:
            show_display: True=显示 OpenCV 检测框叠加窗口（调试用，有内存泄漏）
        """
        log.info("=" * 60)
        log.info("AUTO MODE — full automatic capture with live display")
        if show_display:
            log.info("  OpenCV overlay window ENABLED (detection boxes visible)")
        log.info("  F3: pause/resume")
        log.info("  F4: exit")
        log.info("  Ctrl+C: emergency stop")
        log.info("=" * 60)

        # 检查 USB gadget (只需鼠标；节点可能是 hidg1/hidg2，由 controller 自动探测)
        if not list(Path("/dev").glob("hidg*")):
            log.warning("HID device not found! Run: sudo bash setup_gadget.sh rockchip-mouse")
            log.warning("Continuing without input control...")

        # 显示策略：
        #   --display: 只用 OpenCV 窗口（带检测框叠加，方便调试）
        #   默认:      GStreamer 原生窗口（无检测框，但内存稳定）
        use_gst_display = not show_display
        if show_display:
            log.info("Starting capture (OpenCV overlay mode)...")
        else:
            log.info("Starting capture with native GStreamer display...")
        if not self._init_capture(display=use_gst_display):
            return

        # 等待第一帧
        log.info("Waiting for first frame...")
        for i in range(50):
            frame = self._stream.get_frame()
            if frame is not None:
                log.info(f"First frame OK: shape={frame.shape}, mean={frame.mean():.1f}")
                break
            time.sleep(0.05)
        else:
            log.error("No frame received after 2.5s! Check HDMI connection.")
            self._stream.stop()
            return

        if not self._stream.wait_for_fresh_frames(min_frames=3, timeout=5.0):
            log.error("Capture produced only the first frame; HDMI/GStreamer stream is not advancing.")
            self._stream.stop()
            return

        self._init_detector()

        if not self._init_controller():
            log.warning("Running without controller — no input will be sent")

        self._init_pipeline()

        # 将 GStreamer textoverlay 连接到状态机
        if hasattr(self._stream, 'set_status_text'):
            self._pipeline.set_status_callback(self._stream.set_status_text)

        # 热键处理线程
        def _process_hotkeys():
            while not self.stop_event.is_set():
                if self._hotkey_listener:
                    event = self._hotkey_listener.get_event()
                    if event is not None:
                        if event.value == "stop":
                            log.info("Hotkey: STOP")
                            self.stop_event.set()
                            self._pipeline.stop()
                            break
                        elif event.value == "pause":
                            if self._pipeline.fsm.is_paused:
                                log.info("Hotkey: RESUME")
                                self._pipeline.fsm.resume()
                            else:
                                log.info("Hotkey: PAUSE")
                                self._pipeline.fsm.pause()
                        elif event.value == "start":
                            log.info("Hotkey: START")
                            self._pipeline.fsm.start()
                time.sleep(0.05)

        hotkey_thread = threading.Thread(
            target=_process_hotkeys,
            name="HotkeyHandler",
            daemon=True,
        )
        hotkey_thread.start()

        # 主循环 — GStreamer 原生窗口 + 可选 OpenCV 检测框叠加
        try:
            self._pipeline.run(self.stop_event, show_display=show_display)
        except KeyboardInterrupt:
            log.info("Emergency stop! (Ctrl+C)")
            self.stop_event.set()
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
        finally:
            if self._hotkey_listener:
                self._hotkey_listener.stop()
            if self._controller:
                self._controller.close()
            self._stream.stop()
            self._detector.release()
            cv2.destroyAllWindows()

        # 打印统计
        if self._pipeline:
            stats = self._pipeline.stats
            log.info("=" * 40)
            log.info("Session Statistics:")
            for k, v in stats.items():
                log.info(f"  {k}: {v}")
            log.info("=" * 40)

    # ----------------------------------------------------------
    # 清理
    # ----------------------------------------------------------

    def cleanup(self):
        """释放所有资源。"""
        self.stop_event.set()
        if self._stream:
            self._stream.stop()
        if self._detector:
            self._detector.release()
        if self._controller:
            self._controller.close()
        if self._hotkey_listener:
            self._hotkey_listener.stop()


# ============================================================
# 信号处理
# ============================================================

_app_instance: RocoAutoCapture = None


def _signal_handler(signum, frame):
    """SIGINT/SIGTERM 安全退出。"""
    log.info(f"Received signal {signum}, shutting down...")
    if _app_instance:
        _app_instance.cleanup()
    sys.exit(0)


# ============================================================
# 主入口
# ============================================================

def main():
    global _app_instance

    parser = argparse.ArgumentParser(
        description="洛克王国：世界 — 自动捕捉精灵系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py debug     # 显示画面 + 检测框
  python main.py manual    # 检测结果打印到控制台
  python main.py auto      # 全自动模式 (需先配置 USB gadget)
        """
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="debug",
        choices=["debug", "debug-auto", "manual", "auto"],
        help="运行模式: debug (显示画面), debug-auto (显示+自动控制), manual (仅检测), auto (自动捕捉)",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="配置文件路径 (default: config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (default: INFO)",
    )
    parser.add_argument(
        "-d", "--display",
        action="store_true",
        help="auto 模式下显示 OpenCV 检测框叠加窗口（调试用，长时间运行会缓慢泄漏内存）",
    )
    args = parser.parse_args()

    # 设置日志
    config = load_config(args.config)
    log_config = config.get("logging", {})
    log_level = args.log_level if args.log_level != "INFO" else log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "logs/")
    setup_logging(level=log_level, log_dir=log_dir)

    log.info(f"RocoAutoCapture starting in {args.mode.upper()} mode")
    log.info(f"Config: {args.config}")

    # 创建应用
    app = RocoAutoCapture(config_path=args.config)
    _app_instance = app

    # 注册信号处理
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 运行
    try:
        if args.mode == "debug":
            app.run_debug()
        elif args.mode == "debug-auto":
            app.run_debug_auto()
        elif args.mode == "manual":
            app.run_manual()
        elif args.mode == "auto":
            app.run_auto(show_display=args.display)
    finally:
        app.cleanup()
        log.info("RocoAutoCapture exited.")


if __name__ == "__main__":
    main()
