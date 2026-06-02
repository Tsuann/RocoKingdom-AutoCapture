"""
自动化状态机

实现精灵捕捉的完整流程:

    IDLE ──(定时扫描)──> SCAN ──(发现精灵)──> TRACK
      ^                                         │
      │                                    (锁定目标)
      │                                         v
      │                                       AIM ──(瞄准完成)──> THROW
      │                                                           │
      └───────────────────(检查结果)───────────────────────────────┘
                                    VERIFY

支持键盘热键监听 (需 root 或 input 组权限读取 /dev/input 设备)。
"""

import enum
import os
import threading
import time
from typing import Callable, Optional

from utils import log, Timer


# ============================================================
# 状态定义
# ============================================================

class State(enum.Enum):
    IDLE = "idle"            # 待机, 等待启动
    SCAN = "scan"            # 扫描画面中的精灵
    TRACK = "track"          # 追踪锁定目标
    AIM = "aim"              # 瞄准
    THROW = "throw"          # 蓄力丢球
    VERIFY = "verify"        # 验证捕获结果
    PAUSED = "paused"        # 暂停


# 状态转换事件
class Event(enum.Enum):
    START = "start"              # 启动自动捕捉
    SPRITE_FOUND = "found"       # 检测到精灵
    NO_SPRITE = "no_sprite"      # 未检测到精灵
    TARGET_LOCKED = "locked"     # 目标已锁定
    AIM_READY = "aim_ready"      # 瞄准就绪
    THROW_DONE = "throw_done"    # 丢球完成
    CAPTURE_SUCCESS = "success"  # 捕获成功
    CAPTURE_FAILED = "failed"    # 捕获失败
    PAUSE = "pause"              # 暂停
    RESUME = "resume"            # 恢复
    STOP = "stop"                # 停止


# 状态转换表
TRANSITIONS = {
    State.IDLE: {
        Event.START: State.SCAN,
        Event.STOP: State.IDLE,
    },
    State.SCAN: {
        Event.SPRITE_FOUND: State.TRACK,
        Event.NO_SPRITE: State.SCAN,     # 继续扫描
        Event.PAUSE: State.PAUSED,
        Event.STOP: State.IDLE,
    },
    State.TRACK: {
        Event.TARGET_LOCKED: State.AIM,
        Event.NO_SPRITE: State.SCAN,      # 目标丢失, 回到扫描
        Event.PAUSE: State.PAUSED,
        Event.STOP: State.IDLE,
    },
    State.AIM: {
        Event.AIM_READY: State.THROW,
        Event.NO_SPRITE: State.SCAN,      # 瞄准过程中目标消失
        Event.PAUSE: State.PAUSED,
        Event.STOP: State.IDLE,
    },
    State.THROW: {
        Event.THROW_DONE: State.VERIFY,
        Event.PAUSE: State.PAUSED,
        Event.STOP: State.IDLE,
    },
    State.VERIFY: {
        Event.CAPTURE_SUCCESS: State.IDLE,  # 成功, 重新开始扫描
        Event.CAPTURE_FAILED: State.SCAN,   # 失败, 继续扫描
        Event.STOP: State.IDLE,
    },
    State.PAUSED: {
        Event.RESUME: State.SCAN,          # 恢复后重新扫描
        Event.STOP: State.IDLE,
    },
}


# ============================================================
# 状态机
# ============================================================

class CaptureStateMachine:
    """
    精灵捕捉状态机。

    每个状态有对应的回调函数，由外部注入具体实现。
    """

    def __init__(self):
        self._state = State.IDLE
        self._timer = Timer()
        self._lock = threading.Lock()

        # 状态回调: state → callback()
        self._callbacks: dict = {}

        # 统计
        self._transitions_count = 0
        self._capture_attempts = 0
        self._capture_successes = 0

    # ----------------------------------------------------------
    # 回调注册
    # ----------------------------------------------------------

    def on(self, state: State, callback: Callable[[], Optional[Event]]):
        """注册状态执行回调。回调返回下一个触发事件。"""
        self._callbacks[state] = callback

    # ----------------------------------------------------------
    # 状态转换
    # ----------------------------------------------------------

    def transition(self, event: Event) -> State:
        """
        触发状态转换。返回新状态。

        Raises:
            ValueError: 无效的状态转换
        """
        with self._lock:
            old_state = self._state
            new_state = self._get_next(event)

            if new_state == old_state:
                return new_state

            self._state = new_state
            self._transitions_count += 1
            self._timer.reset()

            log.info(f"State: {old_state.value} --[{event.value}]--> "
                     f"{new_state.value}")
            return new_state

    def _get_next(self, event: Event) -> State:
        current_transitions = TRANSITIONS.get(self._state, {})
        if event not in current_transitions:
            log.warning(f"Invalid transition: {self._state.value} + "
                        f"{event.value}")
            return self._state
        return current_transitions[event]

    # ----------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------

    def run(self, stop_event: Optional[threading.Event] = None):
        """
        运行状态机主循环。

        Args:
            stop_event: 线程事件，设置后退出循环
        """
        self._state = State.IDLE
        self._timer.reset()

        log.info("State machine starting...")

        try:
            while stop_event is None or not stop_event.is_set():
                state = self._state

                # 执行当前状态的逻辑
                callback = self._callbacks.get(state)
                if callback is not None:
                    try:
                        event = callback()
                        if event is not None:
                            self.transition(event)
                    except Exception as e:
                        log.error(f"Error in state {state.value}: {e}")
                        # 出错后回到扫描状态
                        self.transition(Event.STOP)
                        self.transition(Event.START)
                else:
                    # 没有注册回调的状态, 短暂休眠
                    time.sleep(0.1)

        except KeyboardInterrupt:
            log.info("State machine interrupted")
        finally:
            log.info(f"State machine stopped. "
                     f"Attempts: {self._capture_attempts}, "
                     f"Successes: {self._capture_successes}")

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def start(self):
        """启动自动化。"""
        return self.transition(Event.START)

    def stop(self):
        """停止。"""
        return self.transition(Event.STOP)

    def pause(self):
        return self.transition(Event.PAUSE)

    def resume(self):
        return self.transition(Event.RESUME)

    def record_throw(self):
        self._capture_attempts += 1

    def record_success(self):
        self._capture_successes += 1

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state not in (State.IDLE, State.PAUSED)

    @property
    def is_paused(self) -> bool:
        return self._state == State.PAUSED

    @property
    def state_time(self) -> float:
        """当前状态持续时间 (秒)。"""
        return self._timer.elapsed

    @property
    def stats(self) -> dict:
        return {
            "state": self._state.value,
            "state_time": self._timer.elapsed,
            "transitions": self._transitions_count,
            "capture_attempts": self._capture_attempts,
            "capture_successes": self._capture_successes,
        }


# ============================================================
# 热键监听器
# ============================================================

class HotkeyListener:
    """
    Linux 键盘事件监听器。

    通过 /dev/input/event* 读取原始键盘事件，
    检测功能键 (F2, F3, F4) 来触发启动/暂停/退出。

    需要 root 权限或 input 组成员。
    """

    # 默认热键 → 事件映射
    DEFAULT_HOTKEYS = {
        "f2": Event.START,   # 启动
        "f3": Event.PAUSE,   # 暂停/恢复
        "f4": Event.STOP,    # 退出
    }

    # evdev 键码
    KEY_CODES = {
        60: "f2", 61: "f3", 62: "f4",
    }

    def __init__(self, hotkeys: Optional[dict] = None):
        self.hotkeys = hotkeys or self.DEFAULT_HOTKEYS.copy()
        self._input_device: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._event_queue: list = []
        self._lock = threading.Lock()

    def find_keyboard(self) -> Optional[str]:
        """查找键盘输入设备。"""
        import glob
        for dev in sorted(glob.glob("/dev/input/event*")):
            try:
                with open(f"/sys/class/input/{os.path.basename(dev)}/device/name") as f:
                    name = f.read().strip().lower()
                if "keyboard" in name or "kbd" in name:
                    return dev
            except Exception:
                pass
        # 回退: 返回第一个 event 设备
        devices = sorted(glob.glob("/dev/input/event*"))
        return devices[0] if devices else None

    def start(self, device: Optional[str] = None):
        """
        启动热键监听后台线程。

        Args:
            device: 输入设备路径 (如 /dev/input/event3), None 自动检测
        """
        self._input_device = device or self.find_keyboard()
        if self._input_device is None:
            log.warning("No keyboard input device found, hotkeys disabled")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="HotkeyListener",
            daemon=True,
        )
        self._thread.start()
        log.info(f"Hotkey listener started on {self._input_device}")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_event(self) -> Optional[Event]:
        """获取并移除队列中的下一个事件 (非阻塞)。"""
        with self._lock:
            if self._event_queue:
                return self._event_queue.pop(0)
            return None

    def _listen_loop(self):
        """后台监听循环，读取 evdev 事件。"""
        import struct
        try:
            fd = open(self._input_device, "rb")
        except PermissionError:
            log.error(f"Permission denied: {self._input_device}")
            log.error("Run: sudo usermod -a -G input $USER; newgrp input")
            return
        except Exception as e:
            log.error(f"Cannot open {self._input_device}: {e}")
            return

        # evdev 事件格式: struct input_event { timeval, type, code, value }
        EVENT_FORMAT = 'llHHI'
        EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

        try:
            while self._running:
                data = fd.read(EVENT_SIZE)
                if len(data) < EVENT_SIZE:
                    break

                tv_sec, tv_usec, ev_type, ev_code, ev_value = \
                    struct.unpack(EVENT_FORMAT, data)

                # EV_KEY = 1, 按键按下 = 1
                if ev_type == 1 and ev_value == 1:  # Key press
                    key_name = self.KEY_CODES.get(ev_code)
                    if key_name and key_name in self.hotkeys:
                        event = self.hotkeys[key_name]
                        with self._lock:
                            self._event_queue.append(event)
                        log.debug(f"Hotkey: {key_name} → {event.value}")

        except Exception as e:
            log.error(f"Hotkey listener error: {e}")
        finally:
            fd.close()


# ============================================================
# 自动化流程构建器
# ============================================================

class CapturePipeline:
    """
    将检测器、追踪器、控制器和状态机组合成完整流水线。

    用法:
        pipeline = CapturePipeline(detector, controller, config)
        pipeline.run()
    """

    def __init__(self, detector, controller, config: dict):
        self.detector = detector
        self.controller = controller
        self.config = config
        self.tracker = None  # 延迟导入

        from tracker import MultiTracker
        trk_cfg = config.get("tracker", {})
        kalman_cfg = trk_cfg.get("kalman", {})
        self.tracker = MultiTracker(
            max_disappeared=trk_cfg.get("max_disappeared", 30),
            process_noise=kalman_cfg.get("process_noise", 0.03),
            measurement_noise=kalman_cfg.get("measurement_noise", 0.1),
        )

        # 自动化配置
        auto_cfg = config.get("automation", {})
        self.scan_interval = auto_cfg.get("scan_interval", 0.5)
        self.max_throw_attempts = auto_cfg.get("max_throw_attempts", 3)
        self.verify_wait = auto_cfg.get("verify_wait", 3.0)
        self.target_priority = auto_cfg.get("target_priority",
                                            ["shiny", "corrupted", "normal"])

        # 状态机
        self.fsm = CaptureStateMachine()

        # 当前目标
        self._current_target: Optional[dict] = None
        self._throw_count = 0

        # 注册状态回调
        self._register_callbacks()

    # ----------------------------------------------------------
    # 状态回调
    # ----------------------------------------------------------

    def _register_callbacks(self):
        self.fsm.on(State.IDLE, self._on_idle)
        self.fsm.on(State.SCAN, self._on_scan)
        self.fsm.on(State.TRACK, self._on_track)
        self.fsm.on(State.AIM, self._on_aim)
        self.fsm.on(State.THROW, self._on_throw)
        self.fsm.on(State.VERIFY, self._on_verify)
        self.fsm.on(State.PAUSED, self._on_paused)

    def _on_idle(self) -> Optional[Event]:
        time.sleep(0.1)
        return None

    def _on_scan(self) -> Optional[Event]:
        """扫描画面中的精灵。"""
        # 从 capture 模块获取帧 (通过外部注入)
        frame = self._frame_provider() if self._frame_provider else None

        if frame is None:
            time.sleep(self.scan_interval)
            return Event.NO_SPRITE

        # 运行检测
        detections = self.detector.detect_sprites(frame)

        if detections:
            # 更新追踪器
            detections = self.tracker.update(detections)

            # 选择最佳目标
            self._current_target = self.tracker.select_target(
                prefer=self.target_priority,
                detections=detections,
            )

            if self._current_target:
                self._throw_count = 0
                log.info(f"Found sprite: {self._current_target.get('class')} "
                         f"conf={self._current_target.get('confidence', 0):.2f}")
                return Event.SPRITE_FOUND

        time.sleep(self.scan_interval)
        return Event.NO_SPRITE

    def _on_track(self) -> Optional[Event]:
        """追踪并锁定目标。"""
        frame = self._frame_provider() if self._frame_provider else None
        if frame is None:
            return Event.NO_SPRITE

        # 持续检测
        detections = self.detector.detect_sprites(frame)
        detections = self.tracker.update(detections)

        # 尝试重新选择目标
        self._current_target = self.tracker.select_target(
            prefer=self.target_priority,
            detections=detections,
        )

        if self._current_target is None:
            return Event.NO_SPRITE

        # 检查追踪稳定性 (至少追踪了几帧)
        tracker_id = self._current_target.get("tracker_id")
        # 简单判断: 如果连续几帧都有追踪结果, 认为锁定

        # 获取追踪位置
        pos = self.tracker.get_target_position()
        if pos is not None:
            return Event.TARGET_LOCKED

        return Event.NO_SPRITE

    def _on_aim(self) -> Optional[Event]:
        """确认目标已锁定，等待稳定后准备丢球。"""
        # 确认追踪器仍然有目标
        pos = self.tracker.get_target_position()
        if pos is None:
            return Event.NO_SPRITE

        # 不做鼠标瞄准（游戏用 R 键自动瞄准），
        # 只做短暂停顿让追踪稳定
        time.sleep(0.15)

        if self._current_target:
            log.info(f"Aiming at sprite: "
                     f"class={self._current_target.get('class', '?')} "
                     f"tracker_id={self._current_target.get('tracker_id', '?')}")

        return Event.AIM_READY

    def _on_throw(self) -> Optional[Event]:
        """长按 R 键丢球（游戏自动瞄准最近精灵）。"""
        self._throw_count += 1
        self.fsm.record_throw()

        log.info(f"Throw attempt {self._throw_count}/{self.max_throw_attempts}")

        if self.controller is None:
            log.warning("Controller unavailable, simulating throw")
            time.sleep(self.throw_hold_time)
            return Event.THROW_DONE

        # 长按 R 键蓄力丢球
        self.controller.throw_ball()

        time.sleep(0.5)  # 等待动画
        return Event.THROW_DONE

    def _on_verify(self) -> Optional[Event]:
        """
        验证捕获结果 — 四步判定。

        1. 模板匹配 → "捕捉成功" UI   → 成功
        2. 模板匹配 → "捕捉失败" UI   → 失败 (可重试)
        3. 精灵仍在画面中              → 失败 (可重试)
        4. 模板匹配 → "战斗界面" UI   → 失败 (进入战斗)
        5. 都不命中 → 默认成功
        """
        time.sleep(self.verify_wait)

        frame = self._frame_provider() if self._frame_provider else None

        if frame is not None:
            # Step 1: 捕捉成功弹窗
            ui_success = self.detector.detect_ui(
                frame, names=["capture_success", "battle_end"]
            )
            if ui_success:
                name = ui_success[0].get("name", "?")
                log.info(f"✅ Capture SUCCESS — detected UI: {name}")
                self.fsm.record_success()
                self._throw_count = 0
                return Event.CAPTURE_SUCCESS

            # Step 2: 捕捉失败弹窗
            ui_fail = self.detector.detect_ui(
                frame, names=["capture_fail"]
            )
            if ui_fail:
                log.info("❌ Capture FAILED — detected UI: capture_fail")
                # 检查是否还能重试
                if self._throw_count < self.max_throw_attempts:
                    log.info(f"   Retrying ({self._throw_count}/{self.max_throw_attempts})...")
                    return Event.CAPTURE_FAILED
                else:
                    log.info(f"   Max attempts ({self.max_throw_attempts}) reached, giving up")
                    self._throw_count = 0
                    return Event.CAPTURE_SUCCESS  # 放弃，回到 scan

            # Step 3: 画面中还有精灵？
            sprites = self.detector.detect_sprites(frame)
            if sprites and self._throw_count < self.max_throw_attempts:
                log.info(f"❌ Sprite still visible, retrying "
                         f"({self._throw_count}/{self.max_throw_attempts})")
                return Event.CAPTURE_FAILED

            # Step 4: 是否进入战斗了？
            ui_battle = self.detector.detect_ui(
                frame, names=["battle_ui", "skill_bar"]
            )
            if ui_battle:
                log.info("⚔️  Battle started — leaving capture flow")
                self._throw_count = 0
                return Event.CAPTURE_SUCCESS  # 回到 scan（后续可扩展战斗逻辑）

        # Step 5: 默认认为成功
        log.info("✅ Capture VERIFY — no failure signal, assuming success")
        self.fsm.record_success()
        self._throw_count = 0
        return Event.CAPTURE_SUCCESS

    def _on_paused(self) -> Optional[Event]:
        time.sleep(0.2)
        return None  # 外部事件 (热键) 来触发恢复

    # ----------------------------------------------------------
    # 帧提供者
    # ----------------------------------------------------------

    _frame_provider: Optional[Callable] = None

    def set_frame_provider(self, provider: Callable):
        """注入帧获取函数。"""
        self._frame_provider = provider

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def run(self, stop_event: Optional[threading.Event] = None):
        """运行自动化流水线。"""
        self.fsm.start()
        self.fsm.run(stop_event)

    def stop(self):
        self.fsm.stop()

    @property
    def stats(self) -> dict:
        return self.fsm.stats


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    from utils import setup_logging, load_config
    setup_logging(level="INFO", log_dir="logs/")

    config = load_config("config.yaml")

    # 仅测试状态机基本逻辑
    fsm = CaptureStateMachine()

    fsm.transition(Event.START)
    assert fsm.state == State.SCAN
    print(f"State after START: {fsm.state}")

    fsm.transition(Event.SPRITE_FOUND)
    assert fsm.state == State.TRACK
    print(f"State after SPRITE_FOUND: {fsm.state}")

    fsm.transition(Event.PAUSE)
    assert fsm.state == State.PAUSED
    print(f"State after PAUSE: {fsm.state}")

    fsm.transition(Event.RESUME)
    assert fsm.state == State.SCAN
    print(f"State after RESUME: {fsm.state}")

    print("All state machine tests passed!")
