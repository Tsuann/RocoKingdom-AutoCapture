# USB OTG 连接指南 — Orange Pi 5 Ultra

## 硬件连接

```
                      HDMI 线
Windows PC ───────────────────────> Orange Pi 5 Ultra HDMI IN

                    USB-A → USB-C 线
Windows PC <────── USB 3.0 蓝色口 B ──── Orange Pi 5 Ultra
  (USB A 口)         (OTG 口)                (空闲的蓝色 USB 3.0)

                    Type-C 电源线
电源适配器 ──────────────────────────> Orange Pi 5 Ultra Type-C (供电)
```

## Orange Pi 5 Ultra 端口一览

| 端口 | 位置 | 用途 |
|------|------|------|
| **Type-C** | 后面板 | 纯供电 (5V/5A)，无数据功能 |
| **USB 3.0 蓝色 A** | 后面板 | 接键盘鼠标 (host) |
| **USB 3.0 蓝色 B** | 后面板 | **OTG 口 — 连接 PC** (peripheral) |
| USB 2.0 黑色 ×2 | 后面板 | 接外设 (host) |

> 📌 **关键**: Orange Pi 5 Ultra 只有 1 个 Type-C 口，用于供电。OTG 功能在 USB 3.0 蓝色口 B 上。
>
> 📌 需要 **USB-A 转 USB-C 线**（如果 PC 端是 USB-A）或 **USB-C 双头线**（如果 PC 端是 USB-C）。

## 步骤

### 1. 物理连接

- **HDMI 线**: 一端接 Windows PC 的 HDMI 输出，另一端接 Orange Pi 的 HDMI IN 口
- **电源**: Type-C 口接电源适配器（必须独立供电，PC USB 供电不足）
- **OTG 数据线**: 一端接 Orange Pi 的 USB 3.0 蓝色口 B（空闲的那个），另一端接 Windows PC 的 USB 口

### 2. 修改基础 DTB（首次必须）

> ⚠️ DT Overlay 方案已被弃用，原因见下方。

Orange Pi 5 Ultra 默认没有 Type-C CC 控制器，USB 3.0 OTG 口默认工作在 host 模式。DT Overlay 因无法彻底删除 `extcon` 属性（设为 `<0>` 会导致 DWC3 驱动异常，UDC 永远 `not attached`），改为**直接修改基础 DTB**：

```bash
cd /home/orangepi/Desktop/rock

# 1. 反编译当前 DTB
dtc -I dtb -O dts -o /tmp/rk3588-opi5-ultra.dts /boot/dtb/rockchip/rk3588-orangepi-5-ultra.dtb

# 2. 编辑 DTS：在 usb@fc000000 节点中
#    - dr_mode = "otg"; → dr_mode = "peripheral";
#    - 删除 extcon = <0x??>; 整行

# 3. 重新编译 DTB
dtc -I dts -O dtb -o /tmp/rk3588-opi5-ultra-new.dtb /tmp/rk3588-opi5-ultra.dts

# 4. 备份并安装
sudo cp /boot/dtb/rockchip/rk3588-orangepi-5-ultra.dtb /boot/dtb/rockchip/rk3588-orangepi-5-ultra.dtb.bak
sudo cp /tmp/rk3588-opi5-ultra-new.dtb /boot/dtb/rockchip/rk3588-orangepi-5-ultra.dtb

# 5. 移除旧的 overlay 配置（如有）
sudo sed -i 's/overlays=otg-peripheral//' /boot/orangepiEnv.txt

# 6. 重启
sudo reboot
```

### 3. 验证连接

重启后在 Orange Pi 上执行:

```bash
# UDC 是否注册
ls /sys/class/udc/
# 应显示: fc000000.usb

# 连接状态（连上 PC 后应为 connected）
cat /sys/class/udc/fc000000.usb/state
# not attached → 未连 PC 或 DTB 修改未生效
# connected   → 连接成功

# 如果仍为 not attached，检查:
# - DTB 修改是否成功:
#     dtc -I dtb -O dts /boot/dtb/rockchip/rk3588-orangepi-5-ultra.dtb | grep -A 23 "usb@fc000000"
#     确认 dr_mode = "peripheral" 且无 extcon 行
# - orangepiEnv.txt 是否已移除 overlays=otg-peripheral:
#     cat /boot/orangepiEnv.txt | grep overlays
```

### 4. 配置 USB HID Gadget

```bash
cd /home/orangepi/Desktop/rock
sudo bash setup_gadget.sh start
```

成功后会显示:
```
[+] Gadget started successfully!
  Keyboard: ..../functions/hid.kbd
  Mouse:    ..../functions/hid.mouse
  Devices:
  /dev/hidg0  (键盘)
  /dev/hidg1  (鼠标)
```

### 5. 验证 HID 设备

Windows PC 应该会识别到新的 USB 设备:
- 设备管理器中会出现新的 "HID Keyboard Device" 和 "HID-compliant mouse"

在 Orange Pi 上测试:

```bash
# 测试键盘 (按一下 W 键)
python3 -c "
from controller import KeyboardController
k = KeyboardController()
k.open()
k.tap('w', 0.1)
k.close()
print('OK')
"

# 测试鼠标 (画圈)
python3 controller.py mouse
```

### 6. 故障排查

**UDC 为空 (fc000000.usb 未出现)**
- 确认 DTB 修改已安装并重启
- 检查 DTB 中 usb@fc000000 节点是否包含 `dr_mode = "peripheral"` 且无 `extcon`
- 查看内核日志: `dmesg | grep -i dwc3`

**UDC 状态始终为 not attached（即使已插线）**
- 这是 extcon 残留导致的经典问题
- `cat /sys/devices/platform/usbdrd3_0/fc000000.usb/udc/fc000000.usb/soft_connect` 如返回"权限不够"，说明驱动已锁死
- 确认 DTB 中 `extcon` 属性被**完全删除**（不是设为 `<0>`）
- 参考 README.md 第 4 步重新修改 DTB

**HID 设备无反应**
- 确认 USB 线连接到正确的 USB 3.0 口（空闲的那个蓝色口）
- 检查 `/dev/hidg*` 是否创建: `ls -la /dev/hidg*`
- 查看内核日志: `dmesg | tail -20`
- 重新配置: `sudo bash setup_gadget.sh restart`

**需要 root 权限**
- 键鼠设备 `/dev/hidg*` 默认为 root 写权限
- 可添加 udev 规则免 sudo:

```bash
sudo tee /etc/udev/rules.d/99-hidg.rules << 'EOF'
KERNEL=="hidg*", MODE="0666"
EOF
sudo udevadm control --reload
sudo udevadm trigger
```
