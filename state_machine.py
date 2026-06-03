"""
自动化状态机

实现精灵捕捉的完整流程:
    IDLE → SCAN → TRACK → AIM → THROW → VERIFY → SCAN (循环)
"""

import ctypes
import enum
import gc
import os
import threading
import time
from typing import Callable, Optional, Tuple

os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")
os.environ.setdefault("QT_FATAL_WARNINGS", "0")

import numpy as np
from utils import log, Timer, distance

# ---- malloc_trim ----
try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_int]
    _malloc_trim.restype = ctypes.c_int
except Exception:
    _malloc_trim = None

def _trim_memory():
    gc.collect()
    if _malloc_trim is not None:
        try: _malloc_trim(0)
        except Exception: pass

# ============================================================
# 状态定义
# ============================================================

class State(enum.Enum):
    IDLE = "idle"; SCAN = "scan"; TRACK = "track"
    AIM = "aim"; THROW = "throw"; VERIFY = "verify"; PAUSED = "paused"

class Event(enum.Enum):
    START = "start"; SPRITE_FOUND = "found"; NO_SPRITE = "no_sprite"
    TARGET_LOCKED = "locked"; AIM_READY = "aim_ready"; THROW_DONE = "throw_done"
    CAPTURE_SUCCESS = "success"; CAPTURE_FAILED = "failed"
    PAUSE = "pause"; RESUME = "resume"; STOP = "stop"

TRANSITIONS = {
    State.IDLE:    {Event.START: State.SCAN, Event.STOP: State.IDLE},
    State.SCAN:    {Event.SPRITE_FOUND: State.TRACK, Event.NO_SPRITE: State.SCAN,
                    Event.PAUSE: State.PAUSED, Event.STOP: State.IDLE},
    State.TRACK:   {Event.TARGET_LOCKED: State.AIM, Event.NO_SPRITE: State.SCAN,
                    Event.PAUSE: State.PAUSED, Event.STOP: State.IDLE},
    State.AIM:     {Event.AIM_READY: State.THROW, Event.THROW_DONE: State.VERIFY,
                    Event.NO_SPRITE: State.SCAN,
                    Event.PAUSE: State.PAUSED, Event.STOP: State.IDLE},
    State.THROW:   {Event.THROW_DONE: State.VERIFY, Event.PAUSE: State.PAUSED,
                    Event.STOP: State.IDLE},
    State.VERIFY:  {Event.CAPTURE_SUCCESS: State.SCAN, Event.CAPTURE_FAILED: State.SCAN,
                    Event.STOP: State.IDLE},
    State.PAUSED:  {Event.RESUME: State.SCAN, Event.STOP: State.IDLE},
}

# ============================================================
# 状态机
# ============================================================

class CaptureStateMachine:
    def __init__(self):
        self._state = State.IDLE
        self._timer = Timer()
        self._lock = threading.Lock()
        self._callbacks: dict = {}
        self._transitions_count = 0
        self._capture_attempts = 0
        self._capture_successes = 0

    def on(self, state: State, callback: Callable[[], Optional[Event]]):
        self._callbacks[state] = callback

    def transition(self, event: Event) -> State:
        with self._lock:
            old = self._state
            new = TRANSITIONS.get(old, {}).get(event, old)
            if new == old: return new
            self._state = new
            self._transitions_count += 1
            self._timer.reset()
            log.info(f"State: {old.value} --[{event.value}]--> {new.value}")
            return new

    def run(self, stop_event: Optional[threading.Event] = None):
        self._timer.reset()
        self._error_count = 0
        self._loop_count = 0
        log.info(f"State machine starting from {self._state.value}...")
        try:
            while stop_event is None or not stop_event.is_set():
                state = self._state
                self._loop_count += 1
                cb = self._callbacks.get(state)
                if cb is not None:
                    try:
                        event = cb()
                        if event is not None:
                            self.transition(event)
                        self._error_count = 0
                    except Exception as e:
                        self._error_count += 1
                        log.error(f"Error in state {state.value}: {e}")
                        import traceback; traceback.print_exc()
                        if self._error_count >= 5:
                            log.error(f"Too many errors, resetting")
                            self._error_count = 0
                            self.transition(Event.STOP)
                        time.sleep(0.5)
                else:
                    time.sleep(0.1)
                if self._loop_count % 5 == 0:
                    _trim_memory()
        except KeyboardInterrupt:
            log.info("State machine interrupted")
        finally:
            log.info(f"State machine stopped. Loops: {self._loop_count}, "
                     f"Attempts: {self._capture_attempts}, "
                     f"Successes: {self._capture_successes}")

    def start(self): return self.transition(Event.START)
    def stop(self): return self.transition(Event.STOP)
    def pause(self): return self.transition(Event.PAUSE)
    def resume(self): return self.transition(Event.RESUME)
    def record_throw(self): self._capture_attempts += 1
    def record_success(self): self._capture_successes += 1

    @property
    def state(self): return self._state
    @property
    def is_running(self): return self._state not in (State.IDLE, State.PAUSED)
    @property
    def is_paused(self): return self._state == State.PAUSED
    @property
    def state_time(self): return self._timer.elapsed
    @property
    def stats(self): return {
        "state": self._state.value, "state_time": self._timer.elapsed,
        "transitions": self._transitions_count,
        "capture_attempts": self._capture_attempts,
        "capture_successes": self._capture_successes,
    }

# ============================================================
# 热键监听器
# ============================================================

class HotkeyListener:
    KEY_CODES = {60: "f2", 61: "f3", 62: "f4"}

    def __init__(self, hotkeys: Optional[dict] = None):
        self.hotkeys = hotkeys or {"f2": Event.START, "f3": Event.PAUSE, "f4": Event.STOP}
        self._input_device = None
        self._thread = None
        self._running = False
        self._event_queue = []
        self._lock = threading.Lock()

    def find_keyboard(self) -> Optional[str]:
        import glob
        for dev in sorted(glob.glob("/dev/input/event*")):
            try:
                with open(f"/sys/class/input/{os.path.basename(dev)}/device/name") as f:
                    name = f.read().strip().lower()
                if "keyboard" in name or "kbd" in name: return dev
            except Exception: pass
        devices = sorted(glob.glob("/dev/input/event*"))
        return devices[0] if devices else None

    def start(self, device: Optional[str] = None):
        self._input_device = device or self.find_keyboard()
        if self._input_device is None:
            log.warning("No keyboard input device found, hotkeys disabled"); return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, name="Hotkey", daemon=True)
        self._thread.start()
        log.info(f"Hotkey listener started on {self._input_device}")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=2.0)

    def get_event(self) -> Optional[Event]:
        with self._lock:
            return self._event_queue.pop(0) if self._event_queue else None

    def _listen_loop(self):
        import struct
        try: fd = open(self._input_device, "rb")
        except Exception as e: log.error(f"Cannot open {self._input_device}: {e}"); return
        fmt = 'llHHI'; sz = struct.calcsize(fmt)
        try:
            while self._running:
                data = fd.read(sz)
                if len(data) < sz: break
                tv_sec, tv_usec, ev_type, ev_code, ev_value = struct.unpack(fmt, data)
                if ev_type == 1 and ev_value == 1:
                    key_name = self.KEY_CODES.get(ev_code)
                    if key_name and key_name in self.hotkeys:
                        event = self.hotkeys[key_name]
                        with self._lock: self._event_queue.append(event)
                        log.info(f"🔑 Hotkey: {key_name} → {event.value}")
        except Exception as e: log.error(f"Hotkey listener error: {e}")
        finally: fd.close()

# ============================================================
# 自动化流水线
# ============================================================

class CapturePipeline:
    def __init__(self, detector, controller, config: dict):
        self.detector = detector
        self.controller = controller
        self.config = config

        from tracker import MultiTracker
        trk_cfg = config.get("tracker", {})
        kf = trk_cfg.get("kalman", {})
        self.tracker = MultiTracker(
            max_disappeared=trk_cfg.get("max_disappeared", 30),
            process_noise=kf.get("process_noise", 0.03),
            measurement_noise=kf.get("measurement_noise", 0.1),
        )

        auto_cfg = config.get("automation", {})
        self.scan_interval = auto_cfg.get("scan_interval", 0.5)
        self.max_throw_attempts = auto_cfg.get("max_throw_attempts", 3)
        self.verify_wait = auto_cfg.get("verify_wait", 3.0)
        self.target_priority = auto_cfg.get("target_priority", [])

        filt_cfg = auto_cfg.get("detection_filter", {})
        det_cfg = config.get("detector", {})
        self.filter_min_conf = float(filt_cfg.get(
            "min_confidence", det_cfg.get("conf_threshold", 0.15)))
        self.filter_min_area_ratio = float(filt_cfg.get("min_area_ratio", 0.00005))
        self.filter_max_area_ratio = float(filt_cfg.get("max_area_ratio", 0.02))
        self.filter_min_aspect = float(filt_cfg.get("min_aspect", 0.25))
        self.filter_max_aspect = float(filt_cfg.get("max_aspect", 4.0))

        pan_cfg = auto_cfg.get("pan", {})
        self.pan_enabled = pan_cfg.get("enabled", True)
        self.pan_after_empty_scans = pan_cfg.get("empty_scans_before_pan", 5)
        self.pan_direction = pan_cfg.get("direction", "right")
        self.pan_step_pixels = int(pan_cfg.get("step_pixels", 45))
        self.pan_step_delay = float(pan_cfg.get("step_delay", 0.35))
        self.pan_max_steps = int(pan_cfg.get("max_steps", 6))
        self.pan_hold_during_search = bool(pan_cfg.get("hold_during_search", False))
        self._empty_scan_count = 0

        self.fsm = CaptureStateMachine()
        self._current_target: Optional[dict] = None
        self._throw_count = 0
        self._target_position: Optional[Tuple[int, int]] = None
        self._aim_smoothed_position: Optional[Tuple[int, int]] = None

        # 显示
        self._show_display = False
        self._display_win = "RocoAutoCapture — AUTO"
        self._display_fps = 0.0
        self._display_fps_times = []

        # 帧获取
        self._frame_provider: Optional[Callable] = None
        self._last_frame_time = time.perf_counter()
        self._frame_timeout = 5.0

        # 状态回调
        self._status_callback: Optional[Callable[[str], None]] = None

        self._register_callbacks()

    def _filter_detections(self, detections, frame) -> list:
        """自动流程专用过滤：debug 模式保持原始检测，auto 避免锁定离谱误检。"""
        if not detections or frame is None:
            return []

        fh, fw = frame.shape[:2]
        frame_area = max(1, fw * fh)
        filtered = []
        dropped = 0

        for det in detections:
            try:
                x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bw = max(0, min(fw, x2) - max(0, x1))
                bh = max(0, min(fh, y2) - max(0, y1))
                if bw <= 0 or bh <= 0:
                    dropped += 1
                    continue

                conf = float(det.get("confidence", 0.0))
                area_ratio = (bw * bh) / frame_area
                aspect = bw / max(1, bh)

                if conf < self.filter_min_conf:
                    dropped += 1
                    continue
                if not (self.filter_min_area_ratio <= area_ratio <= self.filter_max_area_ratio):
                    dropped += 1
                    continue
                if not (self.filter_min_aspect <= aspect <= self.filter_max_aspect):
                    dropped += 1
                    continue

                filtered.append(det)
            except Exception:
                dropped += 1

        if dropped:
            log.debug(f"Auto detection filter: kept={len(filtered)} dropped={dropped}")
        return filtered

    def _aim_point_from_detection(self, det: Optional[dict]) -> Optional[Tuple[int, int]]:
        """把检测框中心改成更靠上的命中点，默认瞄准头部附近。"""
        if not det:
            return None
        bbox = det.get("bbox")
        if not bbox:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            xr = float(getattr(self.controller, "aim_target_x_ratio", 0.5))
            yr = float(getattr(self.controller, "aim_target_y_ratio", 0.30))
            xr = max(0.0, min(1.0, xr))
            yr = max(0.0, min(1.0, yr))
            bw, bh = x2 - x1, y2 - y1
            smooth = det.get("smoothed_position")
            if smooth is not None:
                sx, sy = smooth
                return (int(sx + (xr - 0.5) * bw), int(sy + (yr - 0.5) * bh))
            return (int(x1 + bw * xr), int(y1 + bh * yr))
        except Exception:
            return None

    def _current_aim_position(self) -> Optional[Tuple[int, int]]:
        raw_pos = self._aim_point_from_detection(self._current_target)
        if raw_pos is None:
            center = self.tracker.get_target_position()
            if center is None:
                return None
            y_offset = int(getattr(self.controller, "aim_fallback_y_offset", -60))
            raw_pos = (int(center[0]), int(center[1] + y_offset))

        smoothing = float(getattr(self.controller, "aim_smoothing", 0.0))
        if smoothing <= 0 or self._aim_smoothed_position is None:
            self._aim_smoothed_position = raw_pos
            return raw_pos

        px, py = self._aim_smoothed_position
        rx, ry = raw_pos
        smoothed = (
            int(px * smoothing + rx * (1.0 - smoothing)),
            int(py * smoothing + ry * (1.0 - smoothing)),
        )
        self._aim_smoothed_position = smoothed
        return smoothed

    def _update_current_target(self, detections) -> None:
        if not detections:
            return
        target = self.tracker.select_target(
            prefer=self.target_priority, detections=detections)
        if target is not None:
            self._current_target = target

    def _detect_and_track(self, frame):
        try:
            detections = self.detector.detect_sprites(frame)
            detections = self._filter_detections(detections, frame)
        except Exception as e:
            log.error(f"Detection error: {e}")
            detections = []
        try:
            tracked = self.tracker.update(detections)
        except Exception as e:
            log.error(f"Tracker error: {e}")
            tracked = detections
        return detections, tracked

    def _select_current_target(self, tracked) -> bool:
        self._current_target = self.tracker.select_target(
            prefer=self.target_priority, detections=tracked)
        if self._current_target is None:
            return False
        self._throw_count = 0
        self._aim_smoothed_position = None
        c = self._current_target.get('class', '?')
        cf = self._current_target.get('confidence', 0)
        log.info(f"Found sprite: {c} conf={cf:.2f}")
        return True

    def _pan_step_until_found(self) -> bool:
        direction = self.pan_direction
        log.info(
            f"Pan search: {direction}, step={self.pan_step_pixels}px, "
            f"max_steps={self.pan_max_steps}, delay={self.pan_step_delay:.2f}s")

        holding = False
        found = False
        try:
            if self.pan_hold_during_search:
                self.controller.mouse.hold("left")
                holding = True
                enter_delay = getattr(self.controller, 'aim_enter_delay', 0.18)
                log.info(f"Pan search: LEFT held, waiting {enter_delay:.2f}s")
                time.sleep(enter_delay)

            for step in range(max(1, self.pan_max_steps)):
                self.controller.pan_view(direction=direction, amount=self.pan_step_pixels)

                time.sleep(self.pan_step_delay)

                frame = self._get_frame()
                if frame is None:
                    continue

                detections, tracked = self._detect_and_track(frame)
                self._show_frame(frame, tracked, "scan")
                self._push_status("scan", tracked)

                if detections and self._select_current_target(tracked):
                    log.info(f"Pan search: target found at step {step + 1}, stopping pan")
                    self._empty_scan_count = 0
                    found = True
                    return True
        except Exception as e:
            log.error(f"Pan error: {e}")
            return False
        finally:
            if holding and not found:
                try:
                    self.controller.mouse.release_button()
                    log.info("Pan search: no target, LEFT released")
                except Exception:
                    pass

        return False

    def _confirm_before_throw(self, scx: int, scy: int,
                              cross_x: int, cross_y: int,
                              pf: float) -> Tuple[bool, int, int]:
        required = int(getattr(self.controller, 'pre_throw_confirmations', 2))
        max_checks = int(getattr(self.controller, 'pre_throw_max_checks', 6))
        interval = float(getattr(self.controller, 'pre_throw_interval', 0.08))
        threshold = float(getattr(self.controller, 'pre_throw_threshold', 14))
        stable = 0

        for i in range(max_checks):
            time.sleep(interval)
            f = self._get_frame()
            if f is not None:
                dets = self.detector.detect_sprites(f)
                dets = self._filter_detections(dets, f)
                tracked = self.tracker.update(dets)
                self._update_current_target(tracked)
                self._show_frame(f, tracked, "aim")
                self._push_status("aim-confirm", tracked)

            cp = self._current_aim_position()
            if cp is None:
                stable = 0
                log.warning(f"AIM confirm[{i + 1}]: target lost")
                continue

            sx, sy = int(cp[0]), int(cp[1])
            cx = scx + cross_x
            cy = scy + cross_y
            ex = sx - cx
            ey = sy - cy
            dist = (ex ** 2 + ey ** 2) ** 0.5
            ey_corr = ey - int(dist * dist * pf / max(1, self.controller.mouse.screen_width))
            err = (ex ** 2 + ey_corr ** 2) ** 0.5

            if err <= threshold:
                stable += 1
                log.info(f"AIM confirm[{i + 1}]: stable {stable}/{required}, err={err:.1f}")
                if stable >= required:
                    return True, cross_x, cross_y
            else:
                stable = 0
                mx_step = getattr(self.controller, 'pid_max_step', 30)
                min_step = getattr(self.controller, 'pid_min_step', 0)
                kP = getattr(self.controller, 'pid_kP', 0.6)
                mx = max(-mx_step, min(mx_step, int(ex * kP)))
                my = max(-mx_step, min(mx_step, int(ey_corr * kP)))
                if abs(mx) < min_step:
                    mx = 0
                if abs(my) < min_step:
                    my = 0
                log.info(f"AIM confirm[{i + 1}]: err={err:.1f}, correcting ({mx},{my})")
                if mx or my:
                    self.controller.mouse.move(mx, my)
                    cross_x += mx
                    cross_y += my

        return False, cross_x, cross_y

    # ---- 帧获取 ----
    def _get_frame(self):
        if self._frame_provider is None: return None
        try:
            frame = self._frame_provider()  # stream.get_frame() 返回拷贝
        except Exception as e:
            log.error(f"Frame provider error: {e}"); return None
        if frame is not None:
            self._last_frame_time = time.perf_counter()
        elif time.perf_counter() - self._last_frame_time > self._frame_timeout:
            log.error(f"No frame for {self._frame_timeout:.0f}s!")
        return frame

    # ---- 显示 ----
    def _init_display(self):
        if not self._show_display: return
        import cv2
        try: cv2.namedWindow(self._display_win, cv2.WINDOW_NORMAL); cv2.resizeWindow(self._display_win, 960, 540)
        except Exception as e: log.warning(f"Cannot create display: {e}"); self._show_display = False

    def _update_fps(self):
        now = time.perf_counter()
        self._display_fps_times.append(now)
        if len(self._display_fps_times) > 15: self._display_fps_times.pop(0)
        if len(self._display_fps_times) >= 2:
            e = self._display_fps_times[-1] - self._display_fps_times[0]
            if e > 0: self._display_fps = (len(self._display_fps_times) - 1) / e

    def _show_frame(self, frame, detections, state_name: str):
        """直接渲染并显示帧（与 debug 模式相同）。"""
        if not self._show_display or frame is None: return
        self._update_fps()
        import cv2
        try:
            from utils import draw_detections
            tid = self._current_target.get("tracker_id") if self._current_target else None
            target_idx = None
            if tid is not None:
                for i, d in enumerate(detections or []):
                    if d.get("tracker_id") == tid: target_idx = i; break
            vis = draw_detections(frame, detections or [], target_idx=target_idx,
                tracker_info={
                    "status": f"State: {state_name} | Backend: {self.detector.backend} | "
                              f"Sprites: {len(detections or [])} | "
                              f"Inf: {self.detector.inference_time*1000:.0f}ms | "
                              f"Throws: {self._throw_count}/{self.max_throw_attempts}",
                    "fps": self._display_fps,
                })
            disp = cv2.resize(vis, (960, 540))
            cv2.imshow(self._display_win, disp)
            cv2.waitKey(1)
        except Exception: pass

    # ---- 状态更新 ----
    def set_status_callback(self, cb): self._status_callback = cb

    def _push_status(self, state_name: str, detections=None):
        if not self._status_callback: return
        target = self._current_target.get('class', '-') if self._current_target else '-'
        be = self.detector.backend.upper()
        im = self.detector.inference_time * 1000
        det_text = ""
        if detections:
            parts = [f"{d.get('class','?')}({sum(d['bbox'][:2])//2},{sum(d['bbox'][2:])//2})={d.get('confidence',0):.2f}"
                     for d in detections[:2]]
            det_text = " | DET: " + ", ".join(parts)
        info = (f"{state_name.upper()} | {be} {im:.0f}ms | Target: {target} | "
                f"Throws: {self._throw_count}/{self.max_throw_attempts}{det_text}")
        try: self._status_callback(info)
        except Exception: pass

    # ---- 注册回调 ----
    def _register_callbacks(self):
        self.fsm.on(State.IDLE, self._on_idle)
        self.fsm.on(State.SCAN, self._on_scan)
        self.fsm.on(State.TRACK, self._on_track)
        self.fsm.on(State.AIM, self._on_aim)
        self.fsm.on(State.THROW, self._on_throw)
        self.fsm.on(State.VERIFY, self._on_verify)
        self.fsm.on(State.PAUSED, self._on_paused)

    def _on_idle(self):
        f = self._get_frame()
        if f is not None: self._show_frame(f, [], "idle")
        time.sleep(0.1)
        return None

    def _on_scan(self):
        frame = self._get_frame()
        if frame is None:
            time.sleep(self.scan_interval)
            return Event.NO_SPRITE

        detections, tracked = self._detect_and_track(frame)

        self._show_frame(frame, tracked, "scan")
        self._push_status("scan", tracked)

        if detections:
            self._empty_scan_count = 0
            if self._select_current_target(tracked):
                return Event.SPRITE_FOUND
        else:
            self._empty_scan_count += 1
            if (self.pan_enabled and self.controller is not None and
                    self._empty_scan_count >= self.pan_after_empty_scans):
                log.info(f"No sprite for {self._empty_scan_count} scans, pan searching...")
                if self._pan_step_until_found():
                    return Event.SPRITE_FOUND
                self._empty_scan_count = 0

        time.sleep(self.scan_interval)
        return Event.NO_SPRITE

    def _on_track(self):
        frame = self._get_frame()
        if frame is None: return Event.NO_SPRITE
        try:
            detections = self.detector.detect_sprites(frame)
            detections = self._filter_detections(detections, frame)
        except Exception as e: log.error(f"Detection error: {e}"); detections = []
        try: detections = self.tracker.update(detections)
        except Exception as e: log.error(f"Tracker error: {e}")
        self._show_frame(frame, detections, "track")
        self._push_status("track", detections)
        self._current_target = self.tracker.select_target(
            prefer=self.target_priority, detections=detections)
        if self._current_target is None:
            return Event.NO_SPRITE
        pos = self._current_aim_position()
        if pos is not None:
            self._target_position = pos
            return Event.TARGET_LOCKED
        return Event.NO_SPRITE

    def _on_aim(self):
        """PID 精确瞄准：按住→蓄力2s→PID循环对齐准星→丢球。"""
        # 持续检测
        frame = self._get_frame()
        if frame is not None:
            try:
                dets = self.detector.detect_sprites(frame)
                dets = self._filter_detections(dets, frame)
                tracked = self.tracker.update(dets)
                self._update_current_target(tracked)
                self._show_frame(frame, tracked, "aim")
                self._push_status("aim", tracked)
            except Exception as e: log.error(f"Detection error: {e}")
        else:
            self._show_frame(None, [], "aim")

        pos = self._current_aim_position()
        if pos is None:
            # 目标丢失时释放按键（必须退出瞄准模式），计入一次丢球
            if self.controller:
                try: self.controller.mouse.release_button()
                except Exception: pass
            self._aim_smoothed_position = None
            log.warning("AIM: target lost, releasing (counts as throw)")
            self._throw_count += 1
            self.fsm.record_throw()
            return Event.THROW_DONE

        self._target_position = pos
        if self.controller is None:
            time.sleep(0.5); return Event.AIM_READY

        try:
            # Step 1: 先按住左键，让游戏进入瞄准/蓄力状态，准星回到中心。
            self.controller.mouse.hold("left")
            enter_delay = getattr(self.controller, 'aim_enter_delay', 0.18)
            log.info(f"AIM: LEFT held — aim mode, waiting {enter_delay:.2f}s")
            time.sleep(enter_delay)

            # Step 2: 蓄力期间持续刷新目标。不要在 hold 前移动到精灵，避免进入瞄准状态时产生偏差。
            charge = getattr(self.controller, 'aim_hold_time', 2.0)
            log.info(f"AIM: charging {charge:.1f}s...")
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < charge:
                time.sleep(getattr(self.controller, 'aim_charge_update_interval', 0.08))
                try:
                    f = self._get_frame()
                    if f is not None:
                        dets = self.detector.detect_sprites(f)
                        dets = self._filter_detections(dets, f)
                        tracked = self.tracker.update(dets)
                        self._update_current_target(tracked)
                        self._show_frame(f, tracked, "aim")
                        self._push_status("aim", tracked)
                except Exception as e:
                    log.error(f"AIM charge error: {e}")
                    import traceback; traceback.print_exc()

            # Step 4: PID 对齐
            sw = self.controller.mouse.screen_width
            sh = self.controller.mouse.screen_height
            scx, scy = sw // 2, sh // 2
            pf = getattr(self.controller, 'parabolic_factor', 0.15)
            thr = getattr(self.controller, 'pid_align_threshold', 8)
            max_it = getattr(self.controller, 'pid_max_iters', 50)
            kP = getattr(self.controller, 'pid_kP', 0.6)
            mx_step = getattr(self.controller, 'pid_max_step', 30)
            min_step = getattr(self.controller, 'pid_min_step', 0)
            pw = getattr(self.controller, 'pid_step_wait', 0.03)

            cross_x, cross_y = 0, 0
            reaim_rounds = getattr(self.controller, 'pre_throw_reaim_rounds', 2)
            pid_detect_interval = getattr(self.controller, 'pid_detect_interval', 1)

            for round_idx in range(reaim_rounds):
                log.info(f"AIM PID round {round_idx + 1}/{reaim_rounds}: kP={kP} thr={thr}px")
                for i in range(max_it):
                    if i % pid_detect_interval == 0:
                        f = self._get_frame()
                        if f is not None:
                            dets = self.detector.detect_sprites(f)
                            dets = self._filter_detections(dets, f)
                            tracked = self.tracker.update(dets)
                            self._update_current_target(tracked)
                            self._show_frame(f, tracked, "aim")
                            self._push_status("aim", tracked)

                    cp = self._current_aim_position()
                    if cp is None:
                        log.warning(f"AIM PID[{i}]: lost")
                        break

                    sx, sy = int(cp[0]), int(cp[1]); self._target_position = cp
                    cx = scx + cross_x; cy = scy + cross_y
                    ex = sx - cx; ey = sy - cy
                    dist = (ex**2 + ey**2)**0.5
                    # 抛物线修正：球下落 ∝ 距离²（而非线性）
                    # pf 已归一化到屏幕宽度，offset = dist² * pf / sw
                    parabolic_offset = int(dist * dist * pf / sw)
                    ey_corr = ey - parabolic_offset
                    err = (ex**2 + ey_corr**2)**0.5

                    if float(err) < thr:
                        log.info(f"AIM PID[{i}]: ALIGNED cursor=({cx},{cy}) sprite=({sx},{sy}) err={err:.1f}")
                        break

                    mx = max(-mx_step, min(mx_step, int(ex * kP)))
                    my = max(-mx_step, min(mx_step, int(ey_corr * kP)))
                    if abs(mx) < min_step:
                        mx = 0
                    if abs(my) < min_step:
                        my = 0

                    if i % 8 == 0 or err < 50:
                        log.info(f"AIM PID[{i}]: cursor=({cx},{cy}) sprite=({sx},{sy}) err=({ex},{ey_corr})→({mx},{my})")

                    if mx or my:
                        self.controller.mouse.move(mx, my)
                        cross_x += mx; cross_y += my
                    time.sleep(pw)
                else:
                    log.warning(f"AIM PID: max iters, final cross=({cross_x},{cross_y})")

                log.info(f"AIM: aligned, cross moved ({cross_x},{cross_y})")
                if getattr(self.controller, 'pre_throw_confirmations', 0) <= 0:
                    log.info("AIM: pre-throw confirmation disabled")
                    break

                ok, cross_x, cross_y = self._confirm_before_throw(
                    scx, scy, cross_x, cross_y, pf)
                if ok:
                    break
                log.warning("AIM: pre-throw confirmation failed, re-aiming")
            else:
                log.warning("AIM: pre-throw confirmation failed, staying in AIM")
                return None
        except Exception as e:
            log.error(f"Aim error: {e}")
            import traceback; traceback.print_exc()

        return Event.AIM_READY

    def _on_throw(self):
        self._throw_count += 1; self.fsm.record_throw()
        self._push_status("throw")
        if self.controller is None: time.sleep(0.5); return Event.THROW_DONE
        try:
            if self.controller.default_aim_mode == "hold_aim":
                log.info(f"Throw #{self._throw_count}: release")
                self.controller.release_throw()
            else:
                pos = self._target_position
                if pos: self.controller.throw_ball_click_mode(pos[0], pos[1])
                else: self.controller.mouse.click("left")
        except Exception as e: log.error(f"Throw error: {e}")
        time.sleep(0.12)
        return Event.THROW_DONE

    def _on_verify(self):
        time.sleep(self.verify_wait)
        frame = self._get_frame()
        self._show_frame(frame, [], "verify")
        self._push_status("verify")
        if frame is not None:
            try:
                ui = self.detector.detect_ui(frame, names=["capture_success", "battle_end"])
                if ui: log.info(f"✅ SUCCESS — {ui[0].get('name','?')}"); self.fsm.record_success(); self._throw_count = 0; return Event.CAPTURE_SUCCESS
                ui = self.detector.detect_ui(frame, names=["capture_fail"])
                if ui: log.info("❌ FAILED"); return Event.CAPTURE_FAILED if self._throw_count < self.max_throw_attempts else self._end_verify()
                sprites = self.detector.detect_sprites(frame)
                sprites = self._filter_detections(sprites, frame)
                if sprites and self._throw_count < self.max_throw_attempts: return Event.CAPTURE_FAILED
                ui = self.detector.detect_ui(frame, names=["battle_ui", "skill_bar"])
                if ui: log.info("⚔️ Battle"); self._throw_count = 0; return Event.CAPTURE_SUCCESS
            except Exception as e: log.error(f"Verify error: {e}")
        return self._end_verify()

    def _end_verify(self):
        log.info("✅ VERIFY — assume success"); self.fsm.record_success(); self._throw_count = 0; return Event.CAPTURE_SUCCESS

    def _on_paused(self):
        self._push_status("paused")
        time.sleep(0.2); return None

    # ---- 生命周期 ----
    def set_frame_provider(self, provider: Callable): self._frame_provider = provider

    def run(self, stop_event=None, show_display=False):
        self._show_display = show_display
        if show_display: self._init_display()
        gs = threading.Event()
        def _gc_loop():
            while not gs.is_set():
                gs.wait(15.0)
                if not gs.is_set():
                    _trim_memory()
        threading.Thread(target=_gc_loop, daemon=True).start()
        try:
            self.fsm.start(); self.fsm.run(stop_event)
        finally:
            gs.set()
            if show_display and self._show_display:
                try: import cv2; cv2.destroyWindow(self._display_win)
                except Exception: pass

    def stop(self):
        if self.controller:
            try: self.controller.mouse.release_button()
            except Exception: pass
        self.fsm.stop()

    @property
    def stats(self): return self.fsm.stats
