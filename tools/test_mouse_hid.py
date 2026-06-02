#!/usr/bin/env python3
"""Manual HID mouse test for movement, click, and long press."""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controller import MouseController, hid_transport_ready
from utils import setup_logging


def find_mouse_hid() -> str:
    """Find the hidg device backing a hid.mouse configfs function."""
    gadget_root = Path("/sys/kernel/config/usb_gadget")
    for dev_file in gadget_root.glob("*/functions/hid.mouse/dev"):
        try:
            major_minor = dev_file.read_text(encoding="utf-8").strip()
            major, minor = major_minor.split(":", 1)
            for hidg in Path("/dev").glob("hidg*"):
                st = os.stat(hidg)
                if os.major(st.st_rdev) == int(major) and os.minor(st.st_rdev) == int(minor):
                    return str(hidg)
        except Exception:
            continue
    return "/dev/hidg1"


def main():
    parser = argparse.ArgumentParser(description="Test HID mouse reports")
    parser.add_argument(
        "device",
        nargs="?",
        default=None,
        help="HID mouse device path. Defaults to auto-detecting hid.mouse",
    )
    args = parser.parse_args()
    device = args.device or find_mouse_hid()

    setup_logging(level="INFO", log_dir="logs/")

    if not hid_transport_ready():
        raise SystemExit("UDC is not configured; run setup_gadget.sh first")

    print(f"using mouse device: {device}")
    mouse = MouseController(device)
    if not mouse.open():
        raise SystemExit(f"Cannot open mouse device: {device}")

    try:
        print("1/4 move right-left")
        for dx, dy in [(180, 0), (-180, 0)]:
            for _ in range(18):
                mouse.move(dx // 18, dy // 18)
                time.sleep(0.025)

        time.sleep(0.5)
        print("2/4 move square")
        for dx, dy in [(160, 0), (0, 120), (-160, 0), (0, -120)]:
            for _ in range(16):
                mouse.move(dx // 16, dy // 16)
                time.sleep(0.025)

        time.sleep(0.5)
        print("3/4 left short click")
        mouse.click("left", duration=0.08)

        time.sleep(0.8)
        print("4/4 left long press, 1.5s")
        mouse.hold_for("left", duration=1.5)

        print("mouse HID test done")
    finally:
        mouse.close()


if __name__ == "__main__":
    main()
