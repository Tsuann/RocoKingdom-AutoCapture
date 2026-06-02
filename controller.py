"""
键鼠控制模块

通过 USB HID Gadget 模拟键盘和鼠标输入。
直接写入 HID 报告到 /dev/hidg0 (键盘) 和 /dev/hidg1 (鼠标)。

HID 报告格式:
- 键盘: [modifier(1), reserved(1), keys(6)] = 8 字节
- 鼠标 (相对): [buttons(1), x(1), y(1), wheel(1)] = 4 字节
  鼠标 (绝对): [buttons(1), x_lo(1), x_hi(1), y_lo(1), y_hi(1), wheel(1)] = 6 字节
"""

import math
import os
import struct
import time
from typing import Optional, Tuple

from utils import log


def hid_transport_ready() -> bool:
    """检查 USB 主机是否已经完成 gadget 枚举配置。"""
    udc_root = "/sys/class/udc"
    try:
        for name in os.listdir(udc_root):
            state_path = os.path.join(udc_root, name, "state")
            with open(state_path, "r", encoding="utf-8") as f:
                state = f.read().strip()
            if state == "configured":
                return True
            log.warning(f"UDC {name} state is {state}, expected configured")
    except Exception as e:
        log.warning(f"Unable to read UDC state: {e}")
    return False


def find_mouse_hid() -> str:
    """在 configfs 中查找 hid.mouse 对应的 /dev/hidg* 设备节点。

    参考 tools/test_mouse_hid.py，用 major:minor 精确匹配，
    因为节点名（/dev/hidg0, /dev/hidg1, /dev/hidg2...）随
    configfs function 创建顺序变化。
    """
    gadget_root = os.path.join("/", "sys", "kernel", "config", "usb_gadget")
    try:
        for func_dir in os.listdir(gadget_root):
            func_path = os.path.join(gadget_root, func_dir, "functions")
            if not os.path.isdir(func_path):
                continue
            mouse_dev = os.path.join(func_path, "hid.mouse", "dev")
            if not os.path.isfile(mouse_dev):
                continue
            with open(mouse_dev, "r", encoding="utf-8") as f:
                major_minor = f.read().strip()
            major, minor = major_minor.split(":", 1)
            for hidg in os.listdir("/dev"):
                if not hidg.startswith("hidg"):
                    continue
                hidg_path = os.path.join("/dev", hidg)
                try:
                    st = os.stat(hidg_path)
                except OSError:
                    continue
                if (os.major(st.st_rdev) == int(major) and
                        os.minor(st.st_rdev) == int(minor)):
                    log.info(f"Auto-detected mouse HID: {hidg_path}")
                    return hidg_path
    except Exception as e:
        log.debug(f"Mouse HID auto-detect failed: {e}")
    return "/dev/hidg1"


# ============================================================
# HID 键码映射
# ============================================================

# USB HID 键盘键码 (部分常用)
KEY_CODES = {
    # 字母
    'a': 0x04, 'b': 0x05, 'c': 0x06, 'd': 0x07, 'e': 0x08,
    'f': 0x09, 'g': 0x0a, 'h': 0x0b, 'i': 0x0c, 'j': 0x0d,
    'k': 0x0e, 'l': 0x0f, 'm': 0x10, 'n': 0x11, 'o': 0x12,
    'p': 0x13, 'q': 0x14, 'r': 0x15, 's': 0x16, 't': 0x17,
    'u': 0x18, 'v': 0x19, 'w': 0x1a, 'x': 0x1b, 'y': 0x1c, 'z': 0x1d,
    # 数字
    '1': 0x1e, '2': 0x1f, '3': 0x20, '4': 0x21,
    '5': 0x22, '6': 0x23, '7': 0x24, '8': 0x25,
    '9': 0x26, '0': 0x27,
    # 功能键
    'enter': 0x28, 'esc': 0x29, 'backspace': 0x2a,
    'tab': 0x2b, 'space': 0x2c,
    'f1': 0x3a, 'f2': 0x3b, 'f3': 0x3c, 'f4': 0x3d,
    'f5': 0x3e, 'f6': 0x3f, 'f7': 0x40, 'f8': 0x41,
    'f9': 0x42, 'f10': 0x43, 'f11': 0x44, 'f12': 0x45,
    # 方向键
    'right': 0x4f, 'left': 0x50, 'down': 0x51, 'up': 0x52,
    # 修饰键
    'shift': 0xe1, 'ctrl': 0xe0, 'alt': 0xe2, 'gui': 0xe3,
    'right_shift': 0xe5, 'right_ctrl': 0xe4, 'right_alt': 0xe6,
}

# 修饰键掩码
MODIFIER_MASK = {
    'ctrl':  0x01,
    'shift': 0x02,
    'alt':   0x04,
    'gui':   0x08,
    'right_ctrl':  0x10,
    'right_shift': 0x20,
    'right_alt':   0x40,
    'right_gui':   0x80,
}

# 鼠标按键掩码
MOUSE_BUTTON = {
    'left':   0x01,
    'right':  0x02,
    'middle': 0x04,
}


# ============================================================
# 键盘控制
# ============================================================

class KeyboardController:
    """USB HID 键盘控制器。"""

    def __init__(self, device_path: str = "/dev/hidg0"):
        self.device_path = device_path
        self._fd: Optional[object] = None
        self._modifiers = 0
        self._keys = [0] * 6

    def open(self) -> bool:
        """打开键盘设备。"""
        if not os.path.exists(self.device_path):
            log.error(f"Keyboard device not found: {self.device_path}")
            log.error("Run: sudo bash setup_gadget.sh start")
            return False

        self._fd = open(self.device_path, "wb")
        log.info(f"Keyboard opened: {self.device_path}")
        return True

    def close(self):
        """释放所有按键并关闭设备。"""
        if self._fd:
            try:
                self.release_all()
            except Exception as e:
                log.debug(f"Keyboard release_all during close failed: {e}")
            try:
                self._fd.close()
            except Exception as e:
                log.debug(f"Keyboard close failed: {e}")
            self._fd = None

    # ----------------------------------------------------------
    # 按键操作
    # ----------------------------------------------------------

    def press(self, key: str):
        """按下按键 (不释放)。"""
        code = self._key_to_code(key)

        if code in (0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7):
            # 修饰键
            mod_bit = self._modifier_bit(key)
            if mod_bit:
                self._modifiers |= mod_bit
        else:
            # 普通键
            if code not in self._keys:
                # 替换第一个空位
                for i in range(6):
                    if self._keys[i] == 0:
                        self._keys[i] = code
                        break
                else:
                    log.warning(f"Keyboard buffer full, cannot press: {key}")

        self._send_report()

    def release(self, key: str):
        """释放按键。"""
        code = self._key_to_code(key)

        if code in (0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7):
            mod_bit = self._modifier_bit(key)
            if mod_bit:
                self._modifiers &= ~mod_bit
        else:
            if code in self._keys:
                self._keys[self._keys.index(code)] = 0

        self._send_report()

    def tap(self, key: str, duration: float = 0.05):
        """按一下并释放。"""
        self.press(key)
        time.sleep(duration)
        self.release(key)

    def hold_for(self, key: str, duration: float = 0.5):
        """按住一段时间后释放 (用于蓄力操作)。"""
        self.press(key)
        time.sleep(duration)
        self.release(key)

    def combo(self, keys: list, duration: float = 0.05):
        """同时按下多个键，然后释放。"""
        for key in keys:
            self.press(key)
        time.sleep(duration)
        for key in reversed(keys):
            self.release(key)

    def release_all(self):
        """释放所有按键。"""
        self._modifiers = 0
        self._keys = [0] * 6
        self._send_report()

    def type_text(self, text: str, interval: float = 0.02):
        """逐字输入文本 (仅支持小写字母和数字)。"""
        for char in text:
            if char == ' ':
                self.tap('space', interval)
            elif char.isupper():
                self.combo(['shift', char.lower()], interval)
            else:
                self.tap(char.lower(), interval)
            time.sleep(interval)

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _send_report(self):
        """发送 HID 键盘报告 (8 字节)。"""
        if self._fd is None:
            return
        report = struct.pack("<BB6B",
                             self._modifiers,  # 修饰键
                             0x00,             # 保留
                             *self._keys)      # 6 个按键
        try:
            self._fd.write(report)
            self._fd.flush()
        except Exception as e:
            log.error(f"Keyboard write error: {e}")

    def _key_to_code(self, key: str) -> int:
        key_lower = key.lower()
        if key_lower in KEY_CODES:
            return KEY_CODES[key_lower]
        log.warning(f"Unknown key: {key}")
        return 0

    def _modifier_bit(self, key: str) -> int:
        key_lower = key.lower()
        for mod_name, mask in MODIFIER_MASK.items():
            if key_lower == mod_name or key_lower == mod_name.replace('_', ''):
                return mask
        return 0

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# 鼠标控制
# ============================================================

class MouseController:
    """USB HID 鼠标控制器 (相对位移模式)。"""

    def __init__(self, device_path: str = None,
                 sensitivity: float = 1.0,
                 screen_width: int = 1920,
                 screen_height: int = 1080):
        self.device_path = device_path or find_mouse_hid()
        self.sensitivity = sensitivity
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._fd: Optional[object] = None
        self._report_length = 4
        self._current_x = screen_width // 2
        self._current_y = screen_height // 2

    def open(self) -> bool:
        """打开鼠标设备。若配置的路径不存在，自动探测。"""
        if not os.path.exists(self.device_path):
            detected = find_mouse_hid()
            if detected != self.device_path and os.path.exists(detected):
                log.info(f"Configured {self.device_path} not found, "
                         f"using auto-detected {detected}")
                self.device_path = detected
            else:
                log.error(f"Mouse device not found: {self.device_path}")
                log.error("Run: sudo bash setup_gadget.sh rockchip-mouse")
                return False

        self._fd = open(self.device_path, "wb")
        self._report_length = self._detect_report_length()
        log.info(f"Mouse opened: {self.device_path}")
        return True

    def close(self):
        if self._fd:
            try:
                self.release_all()
            except Exception as e:
                log.debug(f"Mouse release_all during close failed: {e}")
            try:
                self._fd.close()
            except Exception as e:
                log.debug(f"Mouse close failed: {e}")
            self._fd = None

    # ----------------------------------------------------------
    # 鼠标操作
    # ----------------------------------------------------------

    def move(self, dx: int, dy: int):
        """相对移动鼠标。"""
        # 应用灵敏度和限制范围
        dx = int(dx * self.sensitivity)
        dy = int(dy * self.sensitivity)
        dx = max(-127, min(127, dx))
        dy = max(-127, min(127, dy))

        self._current_x = max(0, min(self.screen_width - 1,
                                     self._current_x + dx))
        self._current_y = max(0, min(self.screen_height - 1,
                                     self._current_y + dy))

        self._send_report(buttons=0, x=dx, y=dy)

    def move_to(self, target_x: int, target_y: int,
                steps: int = 20, step_delay: float = 0.002):
        """
        将鼠标移动到屏幕的绝对坐标位置。
        使用多步平滑移动。

        Args:
            target_x, target_y: 目标坐标 (像素, 相对游戏画面左上角)
            steps: 移动步数
            step_delay: 每步间隔 (秒)
        """
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))

        start_x, start_y = self._current_x, self._current_y
        dx_total = target_x - start_x
        dy_total = target_y - start_y

        if abs(dx_total) < 2 and abs(dy_total) < 2:
            return  # 已经在目标位置

        for i in range(1, steps + 1):
            # 使用 ease-in-out 曲线
            t = i / steps
            ease = t * t * (3 - 2 * t)  # smoothstep

            interp_x = start_x + int(dx_total * ease)
            interp_y = start_y + int(dy_total * ease)

            # 计算这一步的相对位移
            step_dx = interp_x - self._current_x
            step_dy = interp_y - self._current_y

            if step_dx != 0 or step_dy != 0:
                self.move(step_dx, step_dy)

            if step_delay > 0:
                time.sleep(step_delay)

    def click(self, button: str = "left", duration: float = 0.05):
        """点击鼠标按键。"""
        btn_mask = MOUSE_BUTTON.get(button, 0x01)
        self._send_report(buttons=btn_mask)
        time.sleep(duration)
        self._send_report(buttons=0)

    def hold(self, button: str = "left"):
        """按住鼠标按键 (不释放)。"""
        btn_mask = MOUSE_BUTTON.get(button, 0x01)
        self._send_report(buttons=btn_mask)

    def release_button(self):
        """释放所有鼠标按键。"""
        self._send_report(buttons=0)

    def hold_for(self, button: str = "left",
                 duration: float = 0.5):
        """按住鼠标按键一段时间 (用于蓄力丢球等)。"""
        self.hold(button)
        time.sleep(duration)
        self.release_button()

    def scroll(self, amount: int):
        """滚轮滚动 (正=上, 负=下)。"""
        amount = max(-127, min(127, amount))
        self._send_report(buttons=0, wheel=amount)

    def release_all(self):
        """释放所有按键。"""
        self._send_report(buttons=0)

    # ----------------------------------------------------------
    # 高级操作
    # ----------------------------------------------------------

    def aim_and_click(self, target_x: int, target_y: int,
                      button: str = "left",
                      move_steps: int = 15,
                      move_delay: float = 0.002,
                      click_duration: float = 0.05):
        """移动到目标位置并点击。"""
        self.move_to(target_x, target_y, steps=move_steps,
                     step_delay=move_delay)
        time.sleep(0.05)  # 短暂停顿
        self.click(button, duration=click_duration)

    def aim_and_hold(self, target_x: int, target_y: int,
                     hold_duration: float = 0.5,
                     move_steps: int = 15):
        """移动到目标位置并长按 (用于蓄力丢球)。"""
        self.move_to(target_x, target_y, steps=move_steps)
        time.sleep(0.05)
        self.hold_for(button="left", duration=hold_duration)

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _send_report(self, buttons: int = 0, x: int = 0,
                     y: int = 0, wheel: int = 0):
        """发送 HID 鼠标报告 (4 字节, 相对模式)。"""
        if self._fd is None:
            return

        # x, y 转换为有符号字节
        x_byte = x & 0xFF
        y_byte = y & 0xFF
        wheel_byte = wheel & 0xFF

        if self._report_length == 3:
            report = struct.pack("<BBB",
                                 buttons & 0x07,
                                 x_byte,
                                 y_byte)
        else:
            report = struct.pack("<BBBB",
                                 buttons & 0x07,  # 3 个按键
                                 x_byte,
                                 y_byte,
                                 wheel_byte)
        try:
            self._fd.write(report)
            self._fd.flush()
        except Exception as e:
            log.error(f"Mouse write error: {e}")

    def _detect_report_length(self) -> int:
        """从 configfs 读取当前 hidg 设备对应的报告长度。"""
        try:
            dev_stat = os.stat(self.device_path)
            major_minor = f"{os.major(dev_stat.st_rdev)}:{os.minor(dev_stat.st_rdev)}"
            gadget_root = "/sys/kernel/config/usb_gadget"
            for root, dirs, files in os.walk(gadget_root):
                if "dev" not in files or "report_length" not in files:
                    continue
                dev_path = os.path.join(root, "dev")
                with open(dev_path, "r", encoding="utf-8") as f:
                    if f.read().strip() != major_minor:
                        continue
                length_path = os.path.join(root, "report_length")
                with open(length_path, "r", encoding="utf-8") as f:
                    length = int(f.read().strip())
                if length in (3, 4):
                    return length
        except Exception as e:
            log.debug(f"Unable to detect mouse report length: {e}")
        return 4

    @property
    def position(self) -> Tuple[int, int]:
        return (self._current_x, self._current_y)

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# 统一控制器
# ============================================================

class GameController:
    """
    游戏控制器 — 结合键盘和鼠标, 提供高层游戏操作接口。

    用法:
        gc = GameController(config)
        gc.open()
        gc.throw_ball(target_x, target_y, hold_time=500)
        gc.close()
    """

    def __init__(self, config: dict):
        ctrl_cfg = config.get("controller", {})
        self.kbd = KeyboardController(
            device_path=ctrl_cfg.get("keyboard_device", "/dev/hidg0")
        )
        self.mouse = MouseController(
            device_path=ctrl_cfg.get("mouse_device", "/dev/hidg1"),
            sensitivity=ctrl_cfg.get("mouse_sensitivity", 1.0),
        )
        self.throw_hold_time = ctrl_cfg.get("throw_hold_time", 500) / 1000.0
        self.click_delay = ctrl_cfg.get("click_delay", 100) / 1000.0
        self.keymap = ctrl_cfg.get("keymap", {})

        # 鼠标瞄准参数
        aim_cfg = ctrl_cfg.get("aim", {})
        self.aim_move_steps = aim_cfg.get("move_steps", 20)
        self.aim_step_delay = aim_cfg.get("step_delay", 2) / 1000.0
        self.aim_settle_delay = aim_cfg.get("settle_delay", 80) / 1000.0

        # 坐标校准
        calib = ctrl_cfg.get("calibration", {})
        self.calib_offset_x = calib.get("offset_x", 0)
        self.calib_offset_y = calib.get("offset_y", 0)
        self.calib_scale_x = calib.get("scale_x", 1.0)
        self.calib_scale_y = calib.get("scale_y", 1.0)

    def open(self) -> bool:
        """打开键鼠设备。先检查 UDC 是否已连接。"""
        if not hid_transport_ready():
            log.error("UDC not configured — run: sudo bash setup_gadget.sh rockchip-mouse")
            log.error("Then connect USB OTG cable to PC and wait for enumeration.")
            return False

        kbd_ok = self.kbd.open()
        mouse_ok = self.mouse.open()
        if not kbd_ok:
            log.warning("Keyboard unavailable; mouse-only control enabled")
        log.info(f"Mouse device: {self.mouse.device_path}")
        return mouse_ok

    def close(self):
        self.kbd.close()
        self.mouse.close()

    # ----------------------------------------------------------
    # 游戏操作
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # 鼠标瞄准 & 丢球（只用鼠标，不走键盘）
    # ----------------------------------------------------------

    def capture_to_game_coord(self, cap_x: int, cap_y: int) -> Tuple[int, int]:
        """采集画面坐标 → 校准后的游戏屏幕坐标。"""
        gx = int((cap_x - self.calib_offset_x) * self.calib_scale_x)
        gy = int((cap_y - self.calib_offset_y) * self.calib_scale_y)
        return (
            max(0, min(self.mouse.screen_width - 1, gx)),
            max(0, min(self.mouse.screen_height - 1, gy)),
        )

    def aim_at_target(self, target_x: int, target_y: int):
        """平滑移动鼠标光标到精灵位置（采集坐标），自动校准。"""
        gx, gy = self.capture_to_game_coord(target_x, target_y)
        log.info(f"Aim: moving cursor to ({gx}, {gy})")
        self.mouse.move_to(gx, gy,
                           steps=self.aim_move_steps,
                           step_delay=self.aim_step_delay)
        if self.aim_settle_delay > 0:
            time.sleep(self.aim_settle_delay)

    def start_aim(self, target_x: int, target_y: int):
        """按住左键开始瞄准：移动到目标 + 按下左键不松开。"""
        self.aim_at_target(target_x, target_y)
        log.info(f"Aim: holding LEFT button at ({target_x}, {target_y})")
        self.mouse.hold("left")

    def release_throw(self):
        """松开左键 = 丢球！"""
        log.info("Throw: releasing LEFT button!")
        self.mouse.release_button()

    def interact(self):
        """按交互键 (如 W 键与 NPC 对话)。"""
        key = self.keymap.get("interact", "w")
        self.kbd.tap(key, 0.1)

    def run(self, direction: str = "forward"):
        """奔跑 (Shift + 方向键)。"""
        self.kbd.press("shift")
        time.sleep(0.05)
        # 按 W 前进
        self.kbd.tap("w", 0.3)
        time.sleep(0.05)
        self.kbd.release("shift")

    def enter_battle(self):
        """进入战斗后释放技能。"""
        # 按技能键
        skill = self.keymap.get("skill1", "1")
        self.kbd.tap(skill, 0.1)
        time.sleep(0.5)

    def flee(self):
        """逃跑: ESC → 确认。"""
        self.kbd.tap("esc", 0.05)
        time.sleep(0.3)
        self.kbd.tap("enter", 0.05)

    def focus_energy(self):
        """聚能 (X 键)。"""
        key = self.keymap.get("focus", "x")
        self.kbd.tap(key, 0.05)

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def mouse_position(self) -> Tuple[int, int]:
        return self.mouse.position

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# 测试
# ============================================================

def test_mouse_move():
    """测试鼠标移动 (画圈)。"""
    import math
    from utils import setup_logging
    setup_logging(level="INFO", log_dir="logs/")

    mouse = MouseController()
    if not mouse.open():
        log.error("Cannot open mouse device. Is USB gadget set up?")
        log.error("Run: sudo bash setup_gadget.sh start")
        return
    if not hid_transport_ready():
        log.error("USB host has not configured the HID gadget yet.")
        log.error("Reconnect the OTG cable or run: sudo bash setup_gadget.sh restart")
        mouse.close()
        return

    log.info("Moving mouse in a circle... (5 seconds)")

    try:
        center_x, center_y = 960, 540
        radius = 100
        steps = 60

        for i in range(steps):
            angle = (2 * math.pi * i) / steps
            target_x = center_x + int(radius * math.cos(angle))
            target_y = center_y + int(radius * math.sin(angle))
            mouse.move_to(target_x, target_y, steps=5, step_delay=0.005)
            time.sleep(0.02)

        log.info("Test complete")
    finally:
        mouse.close()


def test_keyboard():
    """测试键盘 (按 W 键)。"""
    from utils import setup_logging
    setup_logging(level="INFO", log_dir="logs/")

    kbd = KeyboardController()
    if not kbd.open():
        log.error("Cannot open keyboard device. Is USB gadget set up?")
        return
    if not hid_transport_ready():
        log.error("USB host has not configured the HID gadget yet.")
        log.error("Reconnect the OTG cable or run: sudo bash setup_gadget.sh restart")
        kbd.close()
        return

    log.info("Tapping 'W' key...")
    kbd.tap('w', 0.1)
    log.info("Test complete")
    kbd.close()


if __name__ == "__main__":
    import sys
    from utils import setup_logging
    setup_logging(level="INFO", log_dir="logs/")

    if len(sys.argv) > 1 and sys.argv[1] == "keyboard":
        test_keyboard()
    else:
        test_mouse_move()
