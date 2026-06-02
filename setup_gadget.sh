#!/bin/bash
# ============================================================
# USB HID Gadget 一键配置脚本
# 将 Orange Pi USB 3.0 蓝色口 B (OTG) 配置为 HID 设备 (键盘/鼠标/复合)
#
# 用法: sudo bash setup_gadget.sh [start|stop|restart|status] [both|mouse|keyboard]
# ============================================================

set -e

GADGET_DIR="/sys/kernel/config/usb_gadget/roco_gadget"
UDC=$(ls /sys/class/udc/ 2>/dev/null | head -1)
MODE="${2:-both}"
RECONNECT_DELAY="${RECONNECT_DELAY:-3}"

# USB 描述符
VENDOR_ID="0x2207"       # Rockchip VID, matches the board's known-good gadget
PRODUCT_ID="0x0000"      # Matches the board's known-good gadget descriptor
DEVICE_VER="0x0310"      # Matches the board's known-good gadget descriptor
MANUFACTURER="OrangePi"
PRODUCT="RocoKingdom Capture HID"
SERIAL="$(cat /proc/cpuinfo | grep Serial | cut -d' ' -f2 | tail -c 9)"

# ============================================================
# HID 报告描述符
# ============================================================

# 键盘 HID 报告描述符 (8 字节)
# 1 字节 modifier + 1 字节 reserved + 6 字节按键
KBD_REPORT_DESC="\\x05\\x01\\x09\\x06\\xa1\\x01\\x05\\x07\\x19\\xe0\\x29\\xe7\\x15\\x00\\x25\\x01\\x75\\x01\\x95\\x08\\x81\\x02\\x95\\x01\\x75\\x08\\x81\\x01\\x95\\x05\\x75\\x01\\x05\\x08\\x19\\x01\\x29\\x05\\x91\\x02\\x95\\x01\\x75\\x03\\x91\\x01\\x95\\x6\\x75\\x08\\x15\\x00\\x26\\xff\\x00\\x05\\x07\\x19\\x00\\x29\\xff\\x81\\x00\\xc0"

# 鼠标 HID 报告描述符 (4 字节)
# 1 字节 buttons + 1 字节 x + 1 字节 y + 1 字节 wheel
# 使用相对位移，与 controller.py 的鼠标报告保持一致
MOUSE_REPORT_DESC="\\x05\\x01\\x09\\x02\\xa1\\x01\\x09\\x01\\xa1\\x00\\x05\\x09\\x19\\x01\\x29\\x03\\x15\\x00\\x25\\x01\\x95\\x03\\x75\\x01\\x81\\x02\\x95\\x01\\x75\\x05\\x81\\x03\\x05\\x01\\x09\\x30\\x09\\x31\\x09\\x38\\x15\\x81\\x25\\x7f\\x75\\x08\\x95\\x03\\x81\\x06\\xc0\\xc0"

# ============================================================
# 函数: 解绑占用同一 UDC 的其他 gadget
# ============================================================

unbind_udc_users() {
    if [ -z "$UDC" ]; then
        return
    fi

    for udc_file in /sys/kernel/config/usb_gadget/*/UDC; do
        [ -f "$udc_file" ] || continue

        bound_udc=$(cat "$udc_file" 2>/dev/null || true)
        [ "$bound_udc" = "$UDC" ] || continue
        [ "$udc_file" = "${GADGET_DIR}/UDC" ] && continue

        other_gadget=$(basename "$(dirname "$udc_file")")
        echo "[*] Unbinding existing gadget using $UDC: $other_gadget"
        echo "" > "$udc_file" 2>/dev/null || true
    done
}

unbind_all_udc_users_except() {
    local keep_dir="$1"

    if [ -z "$UDC" ]; then
        return
    fi

    for udc_file in /sys/kernel/config/usb_gadget/*/UDC; do
        [ -f "$udc_file" ] || continue

        bound_udc=$(cat "$udc_file" 2>/dev/null || true)
        [ "$bound_udc" = "$UDC" ] || continue
        [ "$udc_file" = "${keep_dir}/UDC" ] && continue

        other_gadget=$(basename "$(dirname "$udc_file")")
        echo "[*] Unbinding existing gadget using $UDC: $other_gadget"
        echo "" > "$udc_file" 2>/dev/null || true
    done
}

# ============================================================
# 函数: 停止并清理 gadget
# ============================================================

stop_gadget() {
    echo "[*] Stopping gadget..."

    # 解绑 UDC
    if [ -f "${GADGET_DIR}/UDC" ] && [ -n "$(cat ${GADGET_DIR}/UDC 2>/dev/null)" ]; then
        echo "" > "${GADGET_DIR}/UDC" 2>/dev/null || true
    fi

    # 删除配置中的功能链接
    if [ -d "${GADGET_DIR}/configs/b.1" ]; then
        for link in "${GADGET_DIR}/configs/b.1/"*; do
            if [ -L "$link" ]; then
                rm -f "$link" 2>/dev/null || true
            fi
        done
        rmdir "${GADGET_DIR}/configs/b.1/strings/0x409" 2>/dev/null || true
        rmdir "${GADGET_DIR}/configs/b.1" 2>/dev/null || true
    fi

    # 删除功能
    for func in "${GADGET_DIR}/functions/"*; do
        if [ -d "$func" ]; then
            rmdir "$func" 2>/dev/null || true
        fi
    done

    # 删除字符串
    rmdir "${GADGET_DIR}/strings/0x409" 2>/dev/null || true

    # 删除 gadget 目录
    rmdir "${GADGET_DIR}" 2>/dev/null || true

    echo "[+] Gadget stopped."
}

# ============================================================
# 函数: 启动 gadget
# ============================================================

start_gadget() {
    if [ "$MODE" = "mouse" ]; then
        echo "[!] roco_gadget mouse mode is disabled on this board."
        echo "[!] Use: sudo bash setup_gadget.sh rockchip-mouse"
        exit 1
    fi

    echo "[*] Setting up USB HID Gadget..."
    echo "  Mode: $MODE"

    if [ "$MODE" != "both" ] && [ "$MODE" != "mouse" ] && [ "$MODE" != "keyboard" ]; then
        echo "[ERROR] Invalid mode: $MODE"
        echo "Usage: $0 {start|stop|restart|status} {both|mouse|keyboard}"
        exit 1
    fi

    # 检测 UDC
    if [ -z "$UDC" ]; then
        echo "[ERROR] No UDC found! Check USB OTG connection."
        echo "  Available: $(ls /sys/class/udc/ 2>/dev/null || echo 'none')"
        exit 1
    fi
    echo "  UDC: $UDC"

    # 加载模块
    modprobe libcomposite 2>/dev/null || true

    # 释放系统默认 gadget (如 rockchip/ffs.adb) 对同一 UDC 的占用
    unbind_udc_users

    # 停止旧 gadget
    stop_gadget

    echo "[*] Waiting ${RECONNECT_DELAY}s before rebinding UDC..."
    sleep "$RECONNECT_DELAY"

    # 1. 创建 gadget 目录
    mkdir -p "${GADGET_DIR}"
    cd "${GADGET_DIR}"

    case "$MODE" in
        mouse)
            product_name="RocoKingdom Mouse HID"
            config_name="HID Mouse"
            ;;
        keyboard)
            product_name="RocoKingdom Keyboard HID"
            config_name="HID Keyboard"
            ;;
        *)
            product_name="$PRODUCT"
            config_name="HID Composite"
            ;;
    esac

    # 2. 设备描述符
    echo "$VENDOR_ID"  > idVendor
    echo "$PRODUCT_ID" > idProduct
    echo "$DEVICE_VER" > bcdDevice
    echo "0x0210"      > bcdUSB        # Match the known-good rockchip gadget
    echo "0x40"        > bMaxPacketSize0
    echo "super-speed-plus" > max_speed

    # 3. 设备字符串
    mkdir -p strings/0x409
    echo "$MANUFACTURER" > strings/0x409/manufacturer
    echo "$product_name" > strings/0x409/product
    echo "$SERIAL"       > strings/0x409/serialnumber

    # 4. 创建 HID 功能: 键盘
    if [ "$MODE" = "both" ] || [ "$MODE" = "keyboard" ]; then
        mkdir -p functions/hid.kbd
        echo 1  > functions/hid.kbd/protocol        # 1=键盘
        echo 1  > functions/hid.kbd/subclass        # 1=Boot Interface
        echo 8  > functions/hid.kbd/report_length   # 8 字节报告
        echo -ne "$KBD_REPORT_DESC" > functions/hid.kbd/report_desc
    fi

    # 5. 创建 HID 功能: 鼠标
    if [ "$MODE" = "both" ] || [ "$MODE" = "mouse" ]; then
        mkdir -p functions/hid.mouse
        echo 2  > functions/hid.mouse/protocol       # 2=鼠标
        echo 1  > functions/hid.mouse/subclass       # 1=Boot Interface
        echo 1  > functions/hid.mouse/no_out_endpoint
        echo 4  > functions/hid.mouse/report_length  # 4 字节报告
        echo -ne "$MOUSE_REPORT_DESC" > functions/hid.mouse/report_desc
    fi

    # 6. 创建配置
    mkdir -p configs/b.1
    echo 100 > configs/b.1/MaxPower   # 100mA
    mkdir -p configs/b.1/strings/0x409
    echo "$config_name" > configs/b.1/strings/0x409/configuration

    # 7. 链接功能到配置
    if [ "$MODE" = "both" ] || [ "$MODE" = "keyboard" ]; then
        ln -sf functions/hid.kbd configs/b.1/
    fi
    if [ "$MODE" = "both" ] || [ "$MODE" = "mouse" ]; then
        ln -sf functions/hid.mouse configs/b.1/
    fi

    # 8. 绑定 UDC
    echo "$UDC" > UDC

    echo "[+] Gadget started successfully!"
    [ -d "${GADGET_DIR}/functions/hid.kbd" ] && echo "  Keyboard: ${GADGET_DIR}/functions/hid.kbd"
    [ -d "${GADGET_DIR}/functions/hid.mouse" ] && echo "  Mouse:    ${GADGET_DIR}/functions/hid.mouse"
    echo "  Devices:"
    ls -la /dev/hidg* 2>/dev/null || echo "  (hidg devices may take a moment to appear)"
}

# ============================================================
# 函数: 查看状态
# ============================================================

show_status() {
    echo "======== USB Gadget Status ========"
    echo "UDC: $UDC"
    echo ""

    echo "UDC users:"
    for udc_file in /sys/kernel/config/usb_gadget/*/UDC; do
        [ -f "$udc_file" ] || continue
        gadget_name=$(basename "$(dirname "$udc_file")")
        bound_udc=$(cat "$udc_file" 2>/dev/null || true)
        echo "  $gadget_name: ${bound_udc:-unbound}"
    done
    echo ""

    if [ ! -d "$GADGET_DIR" ]; then
        echo "Gadget not configured."
        exit 0
    fi

    if [ -f "${GADGET_DIR}/UDC" ]; then
        bound_udc=$(cat "${GADGET_DIR}/UDC")
        echo "Bound UDC: $bound_udc"
    else
        echo "No UDC bound"
    fi

    echo ""
    echo "HID Devices:"
    ls -la /dev/hidg* 2>/dev/null || echo "  None"
    echo ""
    echo "Kernel messages:"
    dmesg 2>/dev/null | grep -i "gadget\|hidg\|udc" | tail -5 || echo "  None"
}

# ============================================================
# 函数: 使用系统默认 rockchip gadget 承载 HID mouse
# ============================================================

start_rockchip_mouse() {
    local rg="/sys/kernel/config/usb_gadget/rockchip"

    if [ ! -d "$rg" ]; then
        echo "[ERROR] Default rockchip gadget not found: $rg"
        exit 1
    fi
    if [ -z "$UDC" ]; then
        echo "[ERROR] No UDC found!"
        exit 1
    fi

    echo "[*] Reusing default rockchip gadget for HID mouse..."
    echo "  UDC: $UDC"

    unbind_all_udc_users_except "$rg"

    if [ -f "$rg/UDC" ] && [ -n "$(cat "$rg/UDC" 2>/dev/null)" ]; then
        echo "" > "$rg/UDC" 2>/dev/null || true
    fi

    # Remove existing function links from the active configuration, but keep
    # the function directories so system services can restore them later.
    if [ -d "$rg/configs/b.1" ]; then
        for link in "$rg/configs/b.1/"*; do
            [ -L "$link" ] && rm -f "$link"
        done
    fi

    mkdir -p "$rg/functions/hid.mouse"
    echo 2  > "$rg/functions/hid.mouse/protocol"
    echo 1  > "$rg/functions/hid.mouse/subclass"
    echo 1  > "$rg/functions/hid.mouse/no_out_endpoint"
    echo 4  > "$rg/functions/hid.mouse/report_length"
    echo -ne "$MOUSE_REPORT_DESC" > "$rg/functions/hid.mouse/report_desc"

    mkdir -p "$rg/configs/b.1"
    mkdir -p "$rg/configs/b.1/strings/0x409"
    echo "Rockchip HID Mouse" > "$rg/configs/b.1/strings/0x409/configuration"
    ln -sf "$rg/functions/hid.mouse" "$rg/configs/b.1/hid.mouse"

    echo "[*] Waiting ${RECONNECT_DELAY}s before rebinding UDC..."
    sleep "$RECONNECT_DELAY"
    echo "$UDC" > "$rg/UDC"

    echo "[+] rockchip HID mouse started."
    ls -la /dev/hidg* 2>/dev/null || echo "  (hidg devices may take a moment to appear)"
}

# ============================================================
# 主入口
# ============================================================

case "${1:-start}" in
    start)
        start_gadget
        ;;
    stop)
        stop_gadget
        ;;
    restart)
        stop_gadget
        sleep 1
        start_gadget
        ;;
    status)
        show_status
        ;;
    rockchip-mouse)
        start_rockchip_mouse
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|rockchip-mouse} {both|mouse|keyboard}"
        exit 1
        ;;
esac
