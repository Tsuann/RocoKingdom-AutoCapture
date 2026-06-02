"""
USB HID Gadget 管理模块

管理 Orange Pi USB OTG 的 HID gadget 生命周期。
通过 configfs 或 subprocess 调用 setup_gadget.sh。
"""

import os
import subprocess
import time
from typing import Optional

from utils import log

# 默认设备路径 (由 setup_gadget.sh 创建)
DEFAULT_KEYBOARD_DEV = "/dev/hidg0"
DEFAULT_MOUSE_DEV = "/dev/hidg1"

# 脚本路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETUP_SCRIPT = os.path.join(SCRIPT_DIR, "setup_gadget.sh")


class HIDGadget:
    """
    USB HID Gadget 管理器。

    用法:
        gadget = HIDGadget()
        gadget.setup()
        # ... 使用 /dev/hidg0 和 /dev/hidg1 ...
        gadget.teardown()
    """

    def __init__(self,
                 keyboard_dev: str = DEFAULT_KEYBOARD_DEV,
                 mouse_dev: str = DEFAULT_MOUSE_DEV):
        self.keyboard_dev = keyboard_dev
        self.mouse_dev = mouse_dev
        self._active = False

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def setup(self) -> bool:
        """配置并启动 USB HID gadget。需要 sudo。"""
        try:
            result = subprocess.run(
                ["sudo", "bash", SETUP_SCRIPT, "start"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                log.error(f"Gadget setup failed:\n{result.stderr}")
                return False

            log.info("USB HID gadget configured")
            log.debug(f"Setup output:\n{result.stdout}")

            # 等待设备节点出现
            self._wait_for_devices()
            self._active = True
            return True

        except subprocess.TimeoutExpired:
            log.error("Gadget setup timed out")
            return False
        except Exception as e:
            log.error(f"Gadget setup error: {e}")
            return False

    def teardown(self):
        """停止 USB HID gadget。"""
        try:
            subprocess.run(
                ["sudo", "bash", SETUP_SCRIPT, "stop"],
                capture_output=True,
                timeout=10,
            )
            self._active = False
            log.info("USB HID gadget stopped")
        except Exception as e:
            log.error(f"Gadget teardown error: {e}")

    def status(self) -> dict:
        """获取 gadget 状态。"""
        try:
            result = subprocess.run(
                ["sudo", "bash", SETUP_SCRIPT, "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "active": self._active,
                "keyboard_exists": os.path.exists(self.keyboard_dev),
                "mouse_exists": os.path.exists(self.mouse_dev),
                "detail": result.stdout,
            }
        except Exception as e:
            return {
                "active": False,
                "error": str(e),
            }

    # ----------------------------------------------------------
    # 设备文件操作
    # ----------------------------------------------------------

    def open_keyboard(self) -> Optional[object]:
        """打开键盘设备文件 (返回 file object)。"""
        if not os.path.exists(self.keyboard_dev):
            log.error(f"Keyboard device not found: {self.keyboard_dev}")
            return None
        fd = open(self.keyboard_dev, "wb")
        log.debug(f"Opened keyboard: {self.keyboard_dev}")
        return fd

    def open_mouse(self) -> Optional[object]:
        """打开鼠标设备文件 (返回 file object)。"""
        if not os.path.exists(self.mouse_dev):
            log.error(f"Mouse device not found: {self.mouse_dev}")
            return None
        fd = open(self.mouse_dev, "wb")
        log.debug(f"Opened mouse: {self.mouse_dev}")
        return fd

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _wait_for_devices(self, timeout: float = 5.0):
        """等待 /dev/hidg* 设备节点出现。"""
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.keyboard_dev) and os.path.exists(self.mouse_dev):
                log.info("HID gadget devices ready")
                return
            time.sleep(0.2)

        log.warning(f"HID devices not found after {timeout}s. "
                    f"Check OTG cable connection.")

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active and \
               os.path.exists(self.keyboard_dev) and \
               os.path.exists(self.mouse_dev)

    def __del__(self):
        if self._active:
            self.teardown()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.teardown()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    from utils import setup_logging
    setup_logging(level="INFO", log_dir="logs/")

    gadget = HIDGadget()

    print("\nCurrent status:")
    status = gadget.status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # 显示帮助信息
    print(f"""
To set up the USB HID gadget, run:
    sudo bash {SETUP_SCRIPT} start

To stop:
    sudo bash {SETUP_SCRIPT} stop

Make sure:
  1. USB-A to USB-C (or USB-A to USB-A) data cable is connected to the PC
  2. The Orange Pi port is USB 3.0 blue port B, which is configured as OTG peripheral
""")
