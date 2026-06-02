# 洛克王国：世界 — 自动捕捉精灵系统

基于 Orange Pi 5 Ultra（RK3588）的**全自动精灵捕捉**系统。通过 HDMI 采集 PC 游戏画面，使用 YOLO 目标检测 + 卡尔曼追踪定位精灵，再通过 USB HID Gadget 模拟键鼠操作实现全自动捕捉。

```
Windows PC (游戏) ═══ HDMI ═══> Orange Pi 5 Ultra ═══ USB-C OTG ═══> Windows PC (键鼠输入)
                                   │
                                   ├── 画面采集 (HDMI RX)
                                   ├── 精灵检测 (RKNN NPU / ONNX CPU)
                                   ├── 目标追踪 (Kalman + IoU)
                                   └── 状态机控制 (FSM 驱动)
```

---

## 📝 项目约定

> **每次代码/配置修改后必须同步更新本文档。** README 是跨对话会话的"共享记忆"——其他对话实例只能通过 README 了解项目当前状态。若不更新，后续对话会基于过时信息操作，导致重复踩坑。

具体规则：
- 文件路径 / 目录结构变更 → 更新"目录结构"章节
- 配置参数 / 环境变化 → 更新"配置说明"或对应章节
- 踩坑记录 / 故障排查 → 追加到"常见问题"
- 硬件连接 / 端口变化 → 更新"硬件连接"章节
- 新增脚本 / 命令 → 更新相关说明

---

## 目录结构

```
rock/
├── main.py              # 入口脚本，支持 debug / manual / auto 三种模式
├── config.yaml          # 配置文件（模型路径、检测参数、热键等）
├── capture.py           # HDMI 画面采集模块（后台线程抓帧）
├── detector.py          # 精灵检测模块（RKNN / ONNX / 运动检测 + 模板匹配）
├── tracker.py           # 多目标追踪模块（卡尔曼滤波 + IoU 匹配）
├── controller.py        # 键鼠控制模块（USB HID 键盘/鼠标）
├── state_machine.py     # 自动化状态机 + 热键监听器
├── hid_gadget.py        # USB HID Gadget 管理模块
├── utils.py             # 工具函数（日志、坐标变换、FPS 计数、绘制）
├── setup_gadget.sh      # USB HID Gadget 一键配置脚本
├── rk3588-otg-peripheral.dts  # DT overlay 源码（修复 OTG 口设备模式）
├── setup_otg_guide.md   # USB OTG 连接指南
├── requirements.txt     # Python 依赖说明
├── scripts/             # 辅助工具（采集、标注、训练）
│   ├── capture_screenshots.py  # 训练数据截图采集
│   ├── label_tool.py           # 图像标注工具（画边界框）
│   └── train_model.py          # YOLOv8 训练 + ONNX 导出
├── models/              # 模型文件目录（.rknn / .onnx）
├── templates/           # UI 模板图片目录（.png）
└── logs/                # 运行日志
```

---

## 硬件连接

```
                        HDMI 线
Windows PC ──────────────────────────> Orange Pi 5 Ultra HDMI IN

                      USB-A 转 USB-C 线
Windows PC <──────── USB 3.0 蓝色口 B (OTG) ─── Orange Pi 5 Ultra

                      Type-C 电源线
电源适配器 ────────────────────────────> Orange Pi 5 Ultra Type-C (供电)
```

### Orange Pi 5 Ultra 端口说明

| 物理端口 | 控制器 | 角色 | 用途 |
|---------|--------|------|------|
| Type-C | 电源 IC | 纯供电 | 5V/5A 电源输入 |
| **USB 3.0 蓝色口 B** | fc000000.usb | ⚡ OTG → peripheral | **连接 PC，模拟键鼠** |
| USB 3.0 蓝色口 A | fc400000.usb | host（不变）| 接键盘/鼠标 |
| USB 2.0 黑色口 ×2 | fc800000/fc880000 | host（不变）| 接外设 |

> ⚠️ **Orange Pi 5 Ultra 只有 1 个 Type-C 口（纯供电）**。OTG 走的是空闲的 USB 3.0 Type-A 蓝色口。需要 **USB-A 转 USB-C 线**（或 USB-A 双头线）连接 PC。
>
> ⚠️ 修改只影响 USB 3.0 口 B，**其余 3 个 USB-A 口完全不受影响**，键盘鼠标继续正常使用。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | Orange Pi 5 Ultra（或其他 RK3588 开发板带 HDMI IN）|
| 系统 | Ubuntu 22.04（Armbian / Orange Pi 官方镜像）|
| Python | ≥ 3.9 |
| OpenCV | ≥ 4.8（含 GStreamer / V4L2 支持）|
| RKNN | rknn-toolkit-lite2 ≥ 2.0（NPU 加速，可选）|
| ONNX Runtime | ≥ 1.15（CPU fallback，可选）|

已预装的依赖见 `requirements.txt`。

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/orangepi/Desktop/rock
pip install -r requirements.txt
```

### 2. 放置模型文件

将 YOLO 检测模型放入 `models/` 目录：

```bash
# NPU 模型（推荐，~10-30ms 推理）
cp /path/to/sprite_detector.rknn models/

# 或 ONNX 模型（CPU fallback，~100-300ms）
cp /path/to/sprite_detector.onnx models/
```

**如何获取模型？**

社区模型仓库 (`RocoKingdom_AutoCapture`) 暂未发布预训练权重。当前推荐**自行采集数据训练**：

```bash
# Step 1: 采集游戏截图（按 SPACE 保存）
python3 scripts/capture_screenshots.py

# Step 2: 标注精灵边界框（鼠标拖拽画框）
python3 scripts/label_tool.py dataset/raw/

# Step 3: 训练 + 导出 ONNX
python3 scripts/train_model.py --data dataset/labeled/ --epochs 100
```

> 详见 [models/README.md](models/README.md)。

如果没有模型文件，系统会自动使用**运动检测**作为后备方案（精度较低，但可验证流程）。

### 3. 放置 UI 模板（可选）

将游戏 UI 截图放入 `templates/` 目录，用于检测丢球按钮、战斗结束提示等：

| 模板文件 | 作用 |
|---------|------|
| `throw_button.png` | 丢球按钮 |
| `confirm_button.png` | 确认按钮 |
| `battle_end.png` | 战斗结束提示 |
| `ball_icon.png` | 精灵球图标 |

详见 [templates/README.md](templates/README.md)。

### 4. 修复 USB OTG 设备模式（必须）

Orange Pi 5 Ultra 的 USB 3.0 蓝色口 B（fc000000.usb）默认工作在 host 模式，需切换到 **peripheral（设备）模式**。

> ⚠️ **为什么不能用 DT Overlay？** 该板没有 Type-C CC 控制器，extcon 永远不会触发 VBUS 检测。Overlay 若将 extcon 设为 `<0>`（无效 phandle），DWC3 驱动会进入异常状态：UDC 永远 `not attached`，且 `soft_connect` 被内核锁死（返回 EACCES）。Overlay 又无法通过 `/delete-property/` 彻底删除 extcon 属性。
>
> ✅ 因此**直接修改基础 DTB 文件**，彻底删除 extcon 属性 + 设置 `dr_mode = "peripheral"`。

**步骤**：

```bash
cd /home/orangepi/Desktop/rock

# 1. 反编译当前 DTB
dtc -I dtb -O dts -o /tmp/rk3588-opi5-ultra.dts /boot/dtb/rockchip/rk3588-orangepi-5-ultra.dtb

# 2. 编辑 DTS：在 usb@fc000000 节点中
#    - 将 dr_mode = "otg"; 改为 dr_mode = "peripheral";
#    - 删除 extcon = <0x??>; 这一整行

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

重启后验证：

```bash
ls /sys/class/udc/     # 应显示 fc000000.usb
cat /sys/class/udc/fc000000.usb/state  # 连接 PC 后应为 connected（非 not attached）
```

> ✅ 此修改**只影响 USB 3.0 蓝色口 B**，其余 3 个 USB-A 口不受影响。
>
> 📁 DTS overlay 源文件 `rk3588-otg-peripheral.dts` 保留在项目中作为参考，但**不再使用**。

### 5. 连接硬件

1. HDMI 线：PC HDMI 输出 → Orange Pi HDMI IN
2. OTG 数据线：Orange Pi **USB 3.0 蓝色口 B** → PC USB 口（需 USB-A 转 USB-C 或 USB-A 双头线）
3. 电源：Type-C 电源适配器 → Orange Pi Type-C 口（纯供电）

> ⚠️ Orange Pi 5 Ultra 只有 1 个 Type-C 口且**仅用于供电**。OTG 口是后面板的 **USB 3.0 蓝色口 B**，不要插错。

### 6. 运行

```bash
# 调试模式 — 显示画面 + 检测框
python3 main.py debug

# 手动模式 — 只检测，不操作（打印检测结果）
python3 main.py manual

# 全自动模式 — 检测 + 追踪 + 自动操作（需先配置 USB gadget）
sudo bash setup_gadget.sh start
python3 main.py auto
```

---

## 三种运行模式

### Debug 模式

```bash
python3 main.py debug
```

- 显示 960×540 的实时画面窗口
- 画面上叠加检测框（绿色=普通，黄色=异色，红色=污染）
- 当前目标用红色十字准星标记
- **快捷键**：
  - `ESC` — 退出
  - `SPACE` — 保存当前截图 `snapshot_debug.png`
  - `T` — 切换 UI 模板匹配开关

### Manual 模式

```bash
python3 main.py manual
```

- 仅运行检测，**不发送任何键鼠操作**
- 检测结果实时打印到终端
- 适合验证检测器配置和模型精度
- `Ctrl+C` 退出

### Auto 模式（全自动）

```bash
sudo bash setup_gadget.sh start   # 先配置 USB HID
python3 main.py auto              # 再启动自动模式
```

- 运行完整流水线：**检测 → 追踪 → 瞄准 → 丢球 → 验证**
- 由状态机驱动，自动循环
- **快捷键**：
  - `F2` — 启动自动捕捉
  - `F3` — 暂停 / 恢复
  - `F4` — 退出程序
  - `Ctrl+C` — 紧急停止

---

## 配置说明 (`config.yaml`)

<details>
<summary><b>capture</b> — HDMI 画面采集</summary>

```yaml
capture:
  device: 0            # /dev/video 编号
  width: 1920
  height: 1080
  fps: 30
```
</details>

<details>
<summary><b>detector</b> — 精灵检测</summary>

```yaml
detector:
  rknn_model: models/sprite_detector.rknn   # NPU 模型路径
  onnx_model: models/sprite_detector.onnx   # CPU 备选模型
  input_size: [640, 640]                    # YOLO 输入尺寸
  conf_threshold: 0.5                       # 检测置信度阈值
  nms_threshold: 0.4                        # NMS 去重阈值
  classes:
    - normal      # 普通精灵
    - shiny       # 异色精灵（优先捕捉）
    - corrupted   # 污染精灵

  template:
    match_threshold: 0.7      # 模板匹配阈值
    templates_dir: templates/
```

**调参建议**：
- 误检太多 → 提高 `conf_threshold` 到 0.6-0.7
- 漏检太多 → 降低 `conf_threshold` 到 0.3-0.4
- 重复框太多 → 降低 `nms_threshold` 到 0.3
</details>

<details>
<summary><b>tracker</b> — 目标追踪</summary>

```yaml
tracker:
  max_disappeared: 30       # 经过多少帧未检测到后移除追踪器
  kalman:
    process_noise: 0.03     # 越大越相信观测值（响应快但平滑差）
    measurement_noise: 0.1  # 越小越相信观测值
```
</details>

<details>
<summary><b>controller</b> — 鼠标控制（只用鼠标，不走键盘）</summary>

```yaml
controller:
  mouse_sensitivity: 1.0    # 鼠标移动灵敏度
  throw_hold_time: 500      # 长按时间 (ms)
  click_delay: 100          # 点击间隔 (ms)
  keyboard_device: /dev/hidg0
  mouse_device: /dev/hidg1  # 若不存在会自动用 find_mouse_hid() 探测
  keymap:
    interact: w
    throw_ball: r
    run: shift
    skill1: "1"
    skill2: "2"
    escape: esc
    confirm: enter
    focus: x
  aim:
    move_steps: 20          # 鼠标平滑移动步数
    step_delay: 2           # 每步间隔 (ms)
    settle_delay: 80        # 到达后稳定 (ms)
  calibration:              # 采集画面 → 游戏画面坐标校准
    offset_x: 0
    offset_y: 0
    scale_x: 1.0
    scale_y: 1.0
```

> ⚠️ **鼠标设备自动发现**：`mouse_device` 配置的路径如果不存在（如 `/dev/hidg1`），`controller.py` 会通过 `find_mouse_hid()` 在 configfs 中搜索 `hid.mouse` 对应的实际 `/dev/hidg*` 节点。`rockchip-mouse` 模式下节点通常是 `/dev/hidg2`。
</details>

<details>
<summary><b>automation</b> — 自动化状态机</summary>

```yaml
automation:
  scan_interval: 0.5           # 扫描间隔 (秒)
  max_throw_attempts: 3        # 每个目标最多丢球次数
  verify_wait: 3.0             # 丢球后等待验证结果的时间 (秒)
  target_priority:             # 精灵优先级（排前面的优先）
    - shiny
    - corrupted
    - normal
```
</details>

<details>
<summary><b>hotkeys</b> — 热键</summary>

```yaml
hotkeys:
  start: f2
  pause: f3
  exit: f4
```
</details>

---

## 工作原理

### 状态机流程图

```
                      ┌──────────────────────────────────┐
                      │                                  │
   ┌──────┐  START  ┌──────┐  found  ┌───────┐ locked ┌──────┐
   │ IDLE │────────>│ SCAN │────────>│ TRACK │───────>│ AIM  │
   └──────┘         └──────┘         └───────┘        └──────┘
       ↑                │                │      aim_ready  │
       │          no_sprite        no_sprite              ↓
       │                │                │             ┌───────┐
       │                │                │             │ THROW │
       │                │                │             └───────┘
       │                │                │           throw_done
       │          ┌──────────┐          │                ↓
       └──────────│ VERIFY   │<─────────────────  ┌──────────┐
     success/fail │          │   failed/success   │ (wait)   │
                  └──────────┘                    └──────────┘
```

### 检测后端优先级

1. **RKNN YOLO (NPU)** — 推理 ~10-30ms，优先使用
2. **ONNX YOLO (CPU)** — 推理 ~100-300ms，NPU 不可用时自动切换
3. **运动检测** — 无模型兜底方案，基于背景减除 + 帧差法

UI 模板匹配以 OpenCV TM_CCOEFF_NORMED 方式与 YOLO 并行运行。

### 精灵捕捉完整流程

1. **SCAN**：每 0.5s 检测画面中的精灵
2. **TRACK**：卡尔曼滤波追踪精灵位置，处理短暂遮挡
3. **AIM**：**按住鼠标左键** + 移动光标到精灵位置（只用鼠标，不走键盘）
4. **THROW**：**松开左键 = 丢球**，最多尝试 3 次
5. **VERIFY**：四步判定 —
   - 检测"捕捉成功"UI → 成功
   - 检测"捕捉失败"UI → 失败，重试
   - 精灵仍在画面中 → 失败，重试
   - 检测"战斗界面"UI → 进入战斗

> ⚠️ **只用鼠标丢球，键盘不可用。** 游戏只响应鼠标操作，不要使用键盘 R 键等其他方式。

---

## 下一步内容

### 短期（待完善）

- [x] **模型训练/微调** — 训练工具链已完成（`scripts/capture_screenshots.py` + `label_tool.py` + `train_model.py`），**待采集标注数据后实际训练**
- [x] **丢球操作修正** — 按住左键瞄准 + 松开左键丢球（纯鼠标，游戏自动瞄准）
- [x] **捕捉验证增强** — 四步判定：成功UI / 失败UI / 精灵存在 / 战斗UI
- [ ] **模型训练执行** — 采集 50-100+ 张游戏截图并用 label_tool 标注，然后运行 train_model.py
- [ ] **游戏兼容性** — 适配更多分辨率、不同战斗场景
- [ ] **补充 UI 模板** — 截取 capture_success/fail/battle_ui 等模板
- [ ] **战斗逻辑优化** — 加入技能选择、道具使用、逃跑策略
- [ ] **状态机健壮性** — 增加更多错误恢复路径
- [ ] **Web 管理面板** — 远程查看状态、截图、日志

### 长期（设想）

- [ ] **自动寻路** — 结合 OCR / 小地图识别实现自动跑图
- [ ] **精灵仓库管理** — 自动筛选、释放精灵
- [ ] **多开支持** — 同时管理多个游戏窗口

---

## 常见问题 & 注意事项

### USB HID Gadget 相关

| 问题 | 解决方法 |
|------|----------|
| `/dev/hidg*` 不存在 | 检查 OTG 线是否插对接口、是否支持数据传输，运行 `sudo bash setup_gadget.sh start` |
| `echo "$UDC" > UDC` 报“设备或资源忙” | 系统默认 `rockchip` ADB gadget 占用了 `fc000000.usb`。新版 `setup_gadget.sh` 会在启动 HID 前自动解绑占用同一 UDC 的其他 gadget |
| 权限不足 | `sudo usermod -a -G input $USER` 然后重新登录；或添加 udev 规则（见 `setup_otg_guide.md`） |
| UDC 为空 | OTG 线未连接或插错口，`ls /sys/class/udc/` 查看 |
| 按键无反应 | 检查 `dmesg \| tail -20` 查看内核日志 |

### 模型相关

| 问题 | 解决方法 |
|------|----------|
| 没有 RKNN/ONNX 模型 | 系统自动使用运动检测，精度低但可验证流程 |
| RKNN 加载失败 | 可能需要 `sudo` 权限访问 NPU：`sudo python3 main.py debug` |
| 检测卡顿 | 降低 `capture.fps` 或使用 NPU 推理 |

### 运行相关

| 问题 | 解决方法 |
|------|----------|
| HDMI 捕捉失败 | 检查 HDMI 线 + PC 端是否有信号输出，确认 `/dev/video0` 存在 |
| `HotkeyListener` 报权限错误 | `sudo usermod -a -G input $USER` + 重新登录 |
| 鼠标移动位置不准确 | 调整 `mouse_sensitivity` 或确认游戏分辨率为 1920×1080 |
| 丢球总是失败 | 调整 `throw_hold_time` 蓄力时间，或调整 `verify_wait` 等待时间 |

### 其他注意事项

- **NPU 有时需要 sudo**：RK3588 NPU 设备 `/dev/rknpu` 默认可能需要 root 权限
- **HDMI 输入延迟**：RK3588 HDMI RX 采集约有 1-2 帧延迟（~30-60ms），属于正常现象
- **不要同时运行其他 HDMI 采集程序**：`/dev/video0` 只能被一个进程占用
- **WiFi 不稳定**：本设备 WiFi 芯片（Broadcom dhd 驱动）在 RK3588 上偶发断连。建议使用有线网络或锁定 2.4GHz 频段（参考 WiFi 问题排查记录）。
- **安全使用**：本工具仅供学习和研究目的使用，请遵守游戏的相关规定。

---

## 最近测试记录

### 2026-06-02 冒烟测试

在 Orange Pi 当前环境中完成了以下检查：

| 项目 | 结果 |
|------|------|
| Python 语法编译 | ✅ `main.py` / `capture.py` / `detector.py` / `tracker.py` / `controller.py` / `state_machine.py` / `hid_gadget.py` / `utils.py` 通过 |
| 检测器初始化 | ✅ 未放置模型时自动选择 `motion` 后端 |
| 模板加载 | ✅ 当前无 PNG 模板，`detect_ui()` 返回空列表 |
| 状态机基础转换 | ✅ `IDLE → SCAN → TRACK → PAUSED → SCAN` 通过 |
| 键位配置 | ✅ `config.yaml` 已改为 `w` / `r` / `shift` / `1` 等控制器按键名 |
| 运动检测首帧 | ✅ 首帧只初始化背景，不再把整屏误判为目标 |
| HID 鼠标报告 | ✅ `setup_gadget.sh` 已改为 4 字节相对鼠标报告，与 `controller.py` 一致 |
| UDC busy 修复 | ✅ `setup_gadget.sh` 启动前会自动解绑占用 `fc000000.usb` 的默认 `rockchip` gadget |
| Mouse-only HID | ✅ 复用默认 `rockchip` gadget 的 `sudo bash setup_gadget.sh rockchip-mouse` 可进入 `configured` |
| 鼠标写入测试 | ✅ `/dev/hidg1` 左右小幅移动与方形移动测试成功 |
| UDC | ✅ `/sys/class/udc/fc000000.usb` 存在，`rockchip-mouse` 模式下 state 为 `configured` |
| HID 设备节点 | ✅ `rockchip-mouse` 模式下鼠标节点为 `/dev/hidg1` |
| Mouse-only controller | ✅ `GameController.open()` 已允许键盘缺失时使用鼠标控制 |
| HDMI 采集 | ✅ `/dev/video0` 为 `rk_hdmirx`，OpenCV 成功读取 1920×1080 帧 |
| Manual 模式 | ✅ `python3 main.py manual` 可启动采集与 motion 检测 |

### 当前 HID 使用注意事项

- 当前稳定可用的是复用系统默认 gadget 的鼠标模式：

```bash
sudo bash setup_gadget.sh rockchip-mouse
```

- 不要使用 `sudo bash setup_gadget.sh restart mouse`；`roco_gadget` 路径在当前板子上会退回 `default`，Windows 可能显示未知 USB 设备。
- `rockchip-mouse` 模式下鼠标 HID 节点可能是 `/dev/hidg1`、`/dev/hidg2` 等，取决于当前 configfs function 创建顺序。`controller.py` 已内置 `find_mouse_hid()` 自动探测正确节点。
- 鼠标功能测试脚本：

```bash
python3 tools/test_mouse_hid.py
```

- 若脚本报 `UDC is not configured`，先确认：

```bash
cat /sys/class/udc/fc000000.usb/state
bash setup_gadget.sh status
```

- NPU 设备 `/dev/rknpu` 当前尚未完成实测，RKNN 模型推理仍待验证。

### 2026-06-02 鼠标自动丢球更新

| 项目 | 结果 |
|------|------|
| 状态机 IDLE bug 修复 | ✅ `run()` 不再重置状态到 IDLE |
| 鼠标设备自动发现 | ✅ `find_mouse_hid()` 通过 configfs 探测 `/dev/hidg*` |
| UDC 状态检查 | ✅ `GameController.open()` 先检查 `hid_transport_ready()` |
| 鼠标瞄准 | ✅ AIM 状态：按住左键 + 移动光标到精灵位置 |
| 鼠标丢球 | ✅ THROW 状态：松开左键 = 丢球 |
| 键盘丢球 | ❌ 已移除 — 游戏只响应鼠标操作 |
| HDMI 采集 | ✅ GStreamer v4l2src io-mode=4, 1920×1080 |
| OpenCV 显示 | ✅ debug 模式窗口在 GStreamer 前创建，无 Qt/GLib 冲突 |
| motion 检测 + 状态机循环 | ✅ scan → track → aim → throw → verify 完整跑通 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 画面采集 | OpenCV V4L2 + Rockchip HDMI RX |
| 目标检测 | YOLO (RKNN NPU / ONNX CPU) |
| 目标追踪 | Kalman Filter + IoU 匹配 |
| UI 检测 | OpenCV 模板匹配 (TM_CCOEFF_NORMED) |
| 键鼠控制 | Linux USB Gadget (configfs) + HID 报告 |
| 状态管理 | 手动实现的状态机 (FSM) |
| 热键监听 | Linux evdev 设备读取 |

---

## 相关项目

- [RocoKingdom_AutoCapture](https://github.com/ace-trump-tech/RocoKingdom_AutoCapture) — 社区 YOLO 模型来源
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) — Rockchip NPU 工具链
- [USB Gadget ConfigFS](https://www.kernel.org/doc/Documentation/usb/gadget_configfs.txt) — Linux 内核 USB 设备模拟
