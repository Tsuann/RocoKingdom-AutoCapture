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
├── config.yaml          # 配置文件（模型路径、8物种分类、人类化参数等）
├── capture.py           # HDMI 画面采集模块（后台线程抓帧）
├── detector.py          # 精灵检测模块（RKNN / ONNX / 运动检测 + 模板匹配）
├── tracker.py           # 多目标追踪模块（卡尔曼滤波 + IoU 匹配）
├── controller.py        # 键鼠控制模块（人类行为模拟 + 视角旋转 + 长按瞄准）
├── state_machine.py     # 自动化状态机 + 热键监听器 + 视角自动旋转
├── hid_gadget.py        # USB HID Gadget 管理模块
├── utils.py             # 工具函数（日志、坐标变换、FPS 计数、绘制）
├── setup_gadget.sh      # USB HID Gadget 一键配置脚本
├── rk3588-otg-peripheral.dts  # DT overlay 源码（参考，已不使用）
├── setup_otg_guide.md   # USB OTG 连接指南
├── requirements.txt     # Python 依赖说明
├── scripts/             # 辅助工具
│   ├── crawl_sprites.py       # 精灵图片爬虫（百度+Bing+DDG多引擎）
│   ├── extract_frames.py      # 视频帧提取（从录像截帧）
│   ├── auto_label.py          # 自动标注（运动检测+边缘检测）
│   ├── capture_dataset.py     # Windows 截图采集工具
│   ├── label_tool.py          # 图像标注工具（8物种快捷键）
│   ├── augment_dataset.py     # 数据增强（翻转/颜色/噪声/模糊）
│   ├── incremental_train.py   # 增量训练管理（按物种分批）
│   └── train_model.py         # YOLOv8 完整训练 + ONNX 导出
├── dataset/             # 训练数据（PC端）
│   ├── raw_*/           # 原始截图/爬虫图片
│   ├── labeled_*/       # 标注后数据
│   ├── augmented_*/     # 增强后数据
│   └── video/           # 游戏录像
├── models/              # 模型文件
│   ├── sprite_detector.pt    # PyTorch 权重（PC端验证）
│   ├── sprite_detector.onnx  # ONNX 模型（CPU fallback）
│   └── sprite_detector.rknn  # RKNN 模型（NPU 加速，优先使用）
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
# ONNX 模型（CPU 推理，~100-300ms）
cp /path/to/sprite_detector.onnx models/
```

**模型训练（PC端）**

**当前模型**：单类检测（仅 小独角兽），输出 5 通道 `(cx,cy,w,h,obj_conf)`，无分类头。

`config.yaml` 中的 `classes` 需与模型匹配：
```yaml
# 当前单类模型：
classes:
  - xiao_dujiaoshou    # 0: 小独角兽

# 未来多类模型（训练完成后取消注释）：
# classes:
#   - huzhu_quan         # 0: 护主犬
#   - yibei_er           # 1: 伊贝儿
#   - emo_ding           # 2: 恶魔叮
#   - juhua_li           # 3: 菊花梨
#   - gongping_ge        # 4: 公平鸽
#   - ling_hu            # 5: 灵狐
#   - xiao_dujiaoshou    # 6: 小独角兽
#   - xiaoye_yifu        # 7: 小夜/朔夜伊芙
```

| 类别 | 精灵名称 | 状态 |
|------|----------|------|
| 小独角兽 | xiao_dujiaoshou | ✅ V4 昼夜模型，检出率 95%，RKNN+ONNX 已部署 |
| 其他 7 只 | - | 待采集训练 |

**PC端训练流程**：

```bash
# 1. 录视频 — 用 OBS 框选精灵区域录制 2-5 分钟多角度旋转
# 2. 提取帧
python scripts/extract_frames.py dataset/video/xxx.mp4 --sprite xiao_dujiaoshou --fps 2

# 3. 自动标注（运动检测）
python scripts/auto_label.py dataset/raw_xiao_dujiaoshou/ --class-id 6

# 4. 检查标注（可选，用于修正自动标注错误）
python scripts/label_tool.py dataset/labeled_xiao_dujiaoshou/

# 5. 增量训练（5060 Ti 约 10 分钟）
python scripts/incremental_train.py --new-data dataset/labeled_xiao_dujiaoshou/ --device cuda
```

> 详见 [PC训练指南](#pc训练工作流)。如果没有模型文件，系统会自动使用**运动检测**作为后备方案。

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
sudo bash setup_gadget.sh rockchip-mouse
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
sudo bash setup_gadget.sh rockchip-mouse   # 先配置 USB HID（仅鼠标）
python3 main.py auto                        # 再启动自动模式
```

- 运行完整流水线：**检测 → 追踪 → 瞄准 → 丢球 → 验证**
- 由状态机驱动，自动循环
- **画面显示**：GStreamer 原生 `autovideosink` 窗口（1280×720 实时画面）
  - 不使用 OpenCV highgui，避免 Qt/X11 QPixmap 内存泄漏
  - 检测状态和结果通过日志输出（`logs/run_*.log`）
  - Debug 模式下仍可用 OpenCV 窗口查看检测框叠加
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
  fps: 30              # PC 输出 30Hz 时可设 30fps，追求更同步
```
</details>

<details>
<summary><b>detector</b> — 精灵检测</summary>

```yaml
detector:
  rknn_model: models/sprite_detector.rknn   # RKNN 模型路径（NPU，优先）
  onnx_model: models/sprite_detector.onnx   # ONNX 模型路径（CPU fallback）
  input_size: [640, 640]                    # YOLO 输入尺寸
  conf_threshold: 0.16                      # 置信度阈值（太低会产生大量误检）
  nms_threshold: 0.4                        # NMS 去重阈值
  classes:
    - xiao_dujiaoshou                       # 当前单类模型：小独角兽

  template:
    match_threshold: 0.7      # 模板匹配阈值
    templates_dir: templates/
```

**调参建议**：
- 误检太多 → 提高 `conf_threshold` 到 0.15-0.2
- 漏检太多 → 降低 `conf_threshold` 到 0.10-0.12
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
<summary><b>controller</b> — 鼠标控制（人类行为模拟 + 长按瞄准 + 视角旋转）</summary>

```yaml
controller:
  mouse_sensitivity: 1.0    # 鼠标移动灵敏度
  throw_hold_time: 500      # 长按时间 (ms)
  click_delay: 100          # 点击间隔 (ms)

  # 人类行为模拟（避免被检测为脚本）
  humanize: true
  humanize_config:
    jitter_amplitude: 3         # 轨迹噪声（像素）
    overshoot_probability: 0.15 # 过冲概率
    speed_variation: 0.3        # 速度变化幅度

  # 瞄准模式（四步精确瞄准 + PID 对齐 + 抛物线修正）
  aim:
    move_steps: 12              # 平滑移动步数，越少越快
    step_delay: 1               # 每步间隔 (ms)
    settle_delay: 80            # 到达后稳定 (ms)
    hold_time: 1500             # 长按瞄准时间 (ms)，等准心吸附精灵
    enter_delay: 180            # 按住左键后等待准星进入/归中 (ms)
    parabolic_factor: 0.4       # 抛物线修正系数（二次公式: offset = dist² * pf / sw）
    target_x_ratio: 0.5         # bbox 内瞄准点 X：0.5 = 水平中心
    target_y_ratio: 0.30        # bbox 内瞄准点 Y：0.30 = 靠近头部，0.5 = 身体中心
    fallback_y_offset: -60      # 没有 bbox 时，基于追踪中心向上偏移
    charge_update_interval: 0.20 # 蓄力期间重新检测间隔 (s)
    # PID 精确对准参数
    pid_kP: 0.70                # 比例系数（越大越快但可能过冲）
    pid_max_step: 35            # 单步最大移动像素
    pid_min_step: 2             # 小于该像素的修正忽略，减少抖动
    pid_step_wait: 0.02         # 每步间隔 (s)
    pid_align_threshold: 12     # 对齐阈值 (像素)
    pid_max_iters: 70           # 最大迭代次数
    pid_detect_interval: 3      # PID 每多少轮重新检测一次
    aim_smoothing: 0.65         # 瞄准点 EMA 平滑，越大越稳但响应越慢
    pre_throw_confirmations: 0  # 出球前连续确认次数；0 = 关闭
    pre_throw_interval: 0.06    # 出球前确认间隔 (s)
    pre_throw_threshold: 10     # 出球前确认允许误差 (px)
    pre_throw_max_checks: 8     # 最多确认次数；失败则继续 AIM 不松手
    pre_throw_reaim_rounds: 2   # 确认失败后最多追加微调轮数
    default_mode: hold_aim      # hold_aim=长按瞄准+拖拽, click=点击丢球

  # 视角旋转（视野内无精灵时自动旋转）
  pan:
    amount: 100                 # 单次旋转像素量，越小镜头越慢
    default_direction: right    # 默认方向
    alternate: false            # 固定方向搜索，不左右大幅晃动
    scan_width: 240             # 单轮最多旋转宽度

  keyboard_device: /dev/hidg0
  mouse_device: /dev/hidg1
  keymap:
    interact: w
    focus: x
    # ... 其余按键映射
  calibration:              # 采集画面 → 游戏画面坐标校准
    offset_x: 0
    offset_y: 0
    scale_x: 1.0
    scale_y: 1.0
```
</details>

<details>
<summary><b>automation</b> — 自动化状态机</summary>

```yaml
automation:
  scan_interval: 0.35          # 扫描间隔 (秒)
  max_throw_attempts: 3        # 每个目标最多丢球次数
  verify_wait: 1.8             # 丢球后等待验证结果的时间 (秒)

  detection_filter:            # auto/debug-auto 专用，debug 仍显示原始检测
    min_confidence: 0.16        # 保持与 detector.conf_threshold 一致
    min_area_ratio: 0.00012     # 过滤极小噪点
    max_area_ratio: 0.012       # 过滤大面积 UI/背景误检
    min_aspect: 0.30            # bbox 宽高比下限
    max_aspect: 3.2             # bbox 宽高比上限

  target_priority:             # 精灵优先级
    - huzhu_quan
    - yibei_er
    - emo_ding
    - juhua_li
    - gongping_ge
    - ling_hu
    - xiao_dujiaoshou
    - xiaoye_yifu

  pan:                         # 视角自动旋转
    enabled: true              # 是否启用
    empty_scans_before_pan: 5  # 连续空扫描后旋转
    direction: right           # 固定扫描方向
    step_pixels: 80            # 每一步转动像素，越小越稳
    step_delay: 0.15           # 每一步后等待画面稳定 (s)
    max_steps: 6               # 单轮最多扫描步数；检测到精灵会立刻停止
    hold_during_search: true   # 搜索自转时保持长按，发现目标后更快出球
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

1. **SCAN**：每 0.5s 检测画面中的精灵；连续 5 次空扫描 → 自动旋转视角搜索
2. **TRACK**：卡尔曼滤波追踪精灵位置，处理短暂遮挡
3. **AIM**（PID 精确瞄准）：
   - **Step 1**：鼠标移动到精灵位置（告诉游戏瞄准目标）
   - **Step 2**：按住左键进入瞄准模式（准星回到屏幕中心）
   - **Step 3**：2 秒蓄力期间持续检测，更新精灵位置
   - **Step 4**：PID 循环对齐准星到精灵位置，含**二次抛物线修正**（`offset = dist² × pf / sw`，抵消球体飞行下坠）
   - 目标丢失时自动释放按键，计入一次丢球并进入 VERIFY（不再浪费）
4. **THROW**：**松开左键 = 丢球**，最多尝试 3 次
5. **VERIFY**：四步判定 —
   - 检测"捕捉成功"UI → 成功，继续扫描下一个
   - 检测"捕捉失败"UI → 失败，重试
   - 精灵仍在画面中 → 失败，重试
   - 检测"战斗界面"UI → 进入战斗
6. **循环**：成功后自动回到 SCAN 继续搜索（不再卡在 IDLE）

**控制器特性**：
- **人类行为模拟**：鼠标移动加入随机抖动、过冲回正、速度变化
- **PID 精确瞄准**：移动到精灵 → 按住左键 → 2s 蓄力 → PID 循环对齐（含二次抛物线修正）
- **二次抛物线修正**：球体下落 ∝ 距离²，修正公式 `offset = dist² × pf / sw`（`parabolic_factor: 0.4`），近距离不过修、远距离够用
- **鼠标按键追踪**：`move()` 保持当前按键状态，拖拽时不会意外释放左键
- **视角自动旋转**：找不到精灵时左右交替旋转视野

> ⚠️ **只用鼠标丢球，键盘不可用。** 游戏只响应鼠标操作。

### 画面显示

Auto 模式的 GStreamer 原生窗口顶部实时叠加：
```
SCAN | RKNN 18ms | Target: xiao_dujiaoshou | Throws: 1/3 | DET: xiao_dujiaoshou(703,427) 0.11
```
- **STATE**：当前状态机状态（SCAN/TRACK/AIM/THROW/VERIFY/PAUSED）
- **Backend**：推理后端（RKNN/ONNX/motion）+ 推理耗时
- **Target**：当前锁定的精灵种类
- **Throws**：丢球次数 / 最大次数
- **DET**：检测到的精灵名称、中心坐标、置信度

### 内存管理

Auto 模式经过多次优化后运行时内存稳定：
- **GStreamer 原生显示**：使用 `tee → autovideosink` 显示画面，绕过 OpenCV Qt 后端
- **主动 GC**：状态机每 5 次循环 + 每 15 秒定时调用 `gc.collect()` + `malloc_trim(0)`（已修复 GC 循环 bug，确保定时回收执行）
- **队列限制**：GStreamer pipeline 队列 `max-size-buffers=2 leaky=downstream`
- **采集帧率**：降至 10fps（检测只需 ~2fps）
- **显示优化**：先缩放到 960×540 再绘制叠加层，减少每帧内存开销

> ⚠️ **不要使用 Debug 模式的 OpenCV 窗口长时间运行**。OpenCV Qt 后端的 QPixmap 缓存在 X11 中不会释放，约 40MB/s 速度增长，几分钟即可吃满 8GB 内存导致系统卡死。Debug 模式仅用于短时间调试验证。

---

## 下一步内容

### 短期（进行中）

- [x] **模型训练/微调** — 训练工具链已完成，支持增量训练
- [x] **丢球操作修正** — 按住左键瞄准 + 松开左键丢球（纯鼠标）
- [x] **捕捉验证增强** — 四步判定：成功UI / 失败UI / 精灵存在 / 战斗UI
- [x] **人类行为模拟** — 鼠标移动随机抖动/过冲/速度变化
- [x] **视角自动旋转** — 空扫描自动全景搜索
- [x] **小独角兽模型** — mAP50=0.995, ONNX 已导出
- [x] **小独角兽模型 V4** — 昼夜通用，检出率 95% (19/20)，RKNN+ONNX 双后端部署
- [x] **稳定性修复** — 内存稳定 ~160MB，GStreamer 原生显示，Qt 泄漏已隔离
- [x] **四步精确瞄准** — 移动→按住→中心拖拽（抛物线修正）→等待稳定
- [ ] **其他6只精灵** — 录制视频 → 提取帧 → 增量训练
- [ ] **补充 UI 模板** — 截取 capture_success/fail/battle_ui 等模板
- [ ] **游戏兼容性** — 适配更多分辨率、不同战斗场景
- [ ] **状态机健壮性** — 增加更多错误恢复路径
- [ ] **移动目标鲁棒性** — YOLO 检测 + 目标锁定 + 轨迹预测 + 出球时机门控
- [ ] **Web 管理面板** — 远程查看状态、截图、日志

### 后续架构优化方向

当前结论：移动精灵的捕捉准确率不能只靠单帧 YOLO 和 PID 参数解决。识别、连续跟踪、轨迹预测、出球控制需要拆开处理。

推荐路线：

1. **训练数据补强**
   - 增加移动、转身、远近变化、运动模糊、遮挡、昼夜光照变化样本。
   - 不只截取静态清晰帧，要专门采集“正在移动中的精灵”。

2. **目标锁定**
   - YOLO 检到目标后锁定 tracker id。
   - 短暂丢失时不要立刻切换到其他目标，避免多目标场景乱瞄。
   - 目标切换应满足严格条件，例如原目标连续丢失超过 N 帧。

3. **运动检测辅助**
   - 固定场景下可用帧差/光流找移动候选，再用 YOLO 做类别确认。
   - 运动检测不替代 YOLO，只用于缩小候选区域和维持移动目标连续性。

4. **轨迹预测**
   - 使用 Kalman/ByteTrack/OC-SORT 思路，在 YOLO 短暂丢失时预测位置。
   - 出球瞄准未来位置，而不是当前检测点：
     `predicted = current + velocity * lead_time`。
   - `lead_time` 需要覆盖采集延迟、推理延迟、HID 响应、释放前摇和球飞行时间。

5. **出球时机门控**
   - 只有在目标连续可见、tracker 未切换、预测误差稳定、速度可预测时才松手。
   - 如果目标速度过快、方向突变、目标丢失或误差发散，应继续 AIM 或放弃本次。

6. **场景先验**
   - 暂不建议优先做完整 world model。
   - 更实用的是 ROI、屏蔽 UI/天空/水面/边缘区域、限制 bbox 尺寸和出现高度。
   - 固定地图可以作为过滤先验，而不是把整张地图都训练成识别模型。

核心原则：

```text
YOLO 负责识别
Tracker 负责连续性
Kalman/轨迹模型负责短暂丢失和预测
状态机负责判断是否允许出球
PID 只负责把准星移动到目标点
```

### 长期（设想）

- [ ] **自动寻路** — 结合 OCR / 小地图识别实现自动跑图
- [ ] **精灵仓库管理** — 自动筛选、释放精灵
- [ ] **多开支持** — 同时管理多个游戏窗口

---

## 常见问题 & 注意事项

### USB HID Gadget 相关

| 问题 | 解决方法 |
|------|----------|
| `/dev/hidg*` 不存在 | 检查 OTG 线是否插对接口、是否支持数据传输，运行 `sudo bash setup_gadget.sh rockchip-mouse` |
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
| `/dev/video0` 设备忙 | 可能有僵尸进程占用，`ps aux \| grep python.*main` 找到并 `kill -9` |
| `HotkeyListener` 报权限错误 | `sudo usermod -a -G input $USER` + 重新登录 |
| 鼠标移动位置不准确 | 调整 `mouse_sensitivity` 或确认游戏分辨率为 1920×1080 |
| 丢球总是失败 | 调整 `throw_hold_time` 蓄力时间，或调整 `verify_wait` 等待时间 |
| **系统卡死/内存耗尽** | 见下方 "内存问题排查" |
| **画面运行一会后卡住不动** | 已做本轮缓解：appsink 分支按 `capture.fps` 限速、重启加锁；仍需长时间实机验证 |
| **debug-auto 误检很多** | 已加 `automation.detection_filter`，debug-auto 只显示当前帧命中的追踪框 |
| Qt 线程警告 (`QBasicTimer`) | Debug 模式正常。Auto 模式已绕过此问题 |

### 内存问题排查

Auto 模式已优化为内存稳定（~150MB），若遇内存问题：

| 现象 | 可能原因 | 解决方法 |
|------|---------|----------|
| RSS 持续增长（40+ MB/s）| OpenCV Qt 后端的 QPixmap 在 X11 中泄漏 | 不要在 auto 模式中启用 `show_display=True`；auto 模式默认使用 GStreamer 原生显示 |
| 系统卡死不响应 | 内存耗尽触发 OOM Killer 或 swap thrashing | 增大 swap（`sudo fallocate -l 4G /swapfile`），或关闭 Firefox 等大内存应用 |
| NPU 推理后内存增长 | 正常现象，每 5 次循环自动 GC+malloc_trim | 确保未修改 `state_machine.py` 中的 `_trim_memory()` 调用 |
| GStreamer 管道内存 | DMA buffer 在内核中堆积 | 已限制 `max-size-buffers=2`，若仍增长检查 HDMI 信号是否正常 |

**诊断命令**：

```bash
# 查看实时内存
watch -n 1 'ps aux | grep python.*main | grep -v grep'

# 查看内存分布
cat /proc/$(pgrep -f "python.*main.py")/smaps | awk '/^[0-9a-f]/{sz=0}/^Rss:/{sum+=$2}END{print sum/1024 " MB"}'

# 测试：无显示模式确认泄漏源
QT_QPA_PLATFORM=offscreen python3 main.py auto  # 内存稳定 → 泄漏在 Qt/X11
```

### 其他注意事项

- **NPU 可直接使用**：RK3588 NPU 设备 `/dev/rknpu` 当前用户可访问（无需 sudo）
- **NPU 推理 ~20ms**：RKNN 后端比 ONNX CPU（~100-300ms）快约 10 倍
- **HDMI 输入延迟**：RK3588 HDMI RX 采集约有 1-2 帧延迟（~30-60ms），正常现象
- **不要同时运行其他 HDMI 采集程序**：`/dev/video0` 只能被一个进程占用
- **Qt 内存泄漏**：OpenCV 的 Qt5 高 GUI 后端在 X11 下 QPixmap 缓存不释放，**勿用 Debug 模式长时间运行**
- **WiFi 不稳定**：Broadcom dhd 驱动在 RK3588 上偶发断连。建议使用有线网络或锁定 2.4GHz 频段
- **安全使用**：本工具仅供学习和研究目的使用，请遵守游戏的相关规定。

---

## ⚠️ 已知问题：auto/debug-auto 模式卡帧

### 现象

```
python3 main.py debug-auto
```

运行后，**丢完一次球（THROW → VERIFY → SCAN）后画面冻结**：

- OpenCV 显示窗口画面停在最后一帧不再更新
- 系统仍从 `get_frame()` 取到帧（非 None），但**始终是同一帧**
- FSM 基于这帧过期数据继续操作（检测、瞄准、丢球都在错误位置）
- 日志中无 GStreamer 报错，管道名义上仍在 PLAYING 状态

### 已尝试的修复与当前状态

| 尝试 | 结果 |
|------|------|
| 移除 `num-buffers=100`（管道 10 秒自毁）| ❌ 无效 — 管道不设限也会停 |
| 帧队列 2→4（减少帧饥饿）| ❌ 无效 — 不是争抢问题 |
| 显示线程用 `get_frame_nonblock()`（省拷贝）| ❌ 无效 |
| PID 循环每 3 迭代才检测一次（减少帧消耗）| ❌ 无效 |
| `_on_new_sample` 加 try/except（防 GLib 崩）| ❌ 无效 |
| 总线 EOS/ERROR 监听 + `is_stale` 检测 + auto-restart | ❌ 无效 — 管道未报任何错误 |
| `v4l2src` 显式 `io-mode=2`（MMAP）| ❌ 无效 |
| appsink 分支加入 `videorate`，按 `capture.fps` 实际限速 | 🟡 已修改，待实机长时间验证 |
| GStreamer bus watch 停止时释放 + restart 互斥锁 | 🟡 已修改，待实机长时间验证 |
| AIM 阶段 `sw` 未定义修复 | ✅ 已修复，会减少自动流程异常和反复报错 |

### 诊断关键点

两线程架构：**FSM 线程**（状态机 + 检测 + 控制）和**显示线程**（主线程，cv2.imshow 渲染）共享同一个 `HDMIStream` 的帧队列。

关键观察：
- `get_frame()` 持续返回帧（非 None），但始终是同一帧 → 说明 `appsink` 的 `new-sample` 信号已停止触发
- 管道状态仍为 PLAYING，无 EOS/ERROR 消息 → `v4l2src` 没有主动报告错误
- 问题只在**丢球后**出现，而不是一开始就卡 → 与丢球操作（鼠标释放 → HID 写入 /dev/hidg1）有时序关联

### 待排查方向

1. **USB HID 写入阻塞** — `mouse.release_button()` 写 `/dev/hidg*` 时，若 USB host 停止读取 HID 报告，`write()` 可能阻塞。虽然 Linux 文件 I/O 会释放 GIL，但若内核内部有全局锁（如 USB gadget 的 `ep->lock`），可能间接影响其他子系统
2. **HDMI 信号变化** — 游戏在捕捉动画期间可能切换分辨率/刷新率，导致 `rk_hdmirx` 驱动失同步。V4L2 驱动可能进入静默失败状态（buffer 排队但不产出）
3. **CMA/DMABUF 耗尽** — NPU 推理 + 帧采集共用 CMA 堆。AIM 阶段高频 NPU 推理可能耗尽 CMA，导致 `v4l2src` 的 DMABUF 分配静默失败
4. **GLib 主循环线程静默退出** — `_run_loop` 的 `except Exception` 可能没捕获所有错误类型（如 `SystemError`、`GError`）。线程退出后 `is_running` 变 False，但 `_running` 仍为 True
5. **GStreamer appsink 内部队列满** — 若回调 `_on_new_sample` 耗时过长（大量帧拷贝），appsink 内部队列可能满，新 buffer 被 `drop=true` 丢弃。但即使丢弃，new-sample 信号仍应触发
6. **NPU 推理阻塞 GStreamer 线程** — `detect_sprites()` 的 `rknn_run` 是 ctypes 同步调用。虽在 FSM 线程，但若 NPU 驱动有全局互斥，可能阻塞 GLib 主循环中的 DMA buffer 释放

### 建议的调试手段

```bash
# 1. 确认 GLib 主循环线程是否还活着
ps -T -p $(pgrep -f "python.*main.py") | grep HDMI

# 2. 查看 GStreamer 管道实时状态（需 gst-launch 环境）
GST_DEBUG=v4l2src:3 python3 main.py debug-auto 2>&1 | tee gst_debug.log

# 3. 确认 USB HID 写入是否阻塞
# 在 mouse.release_button() 前后打时间戳，看耗时是否异常

# 4. 检查 CMA 使用量
cat /proc/meminfo | grep -i cma

# 5. 确认 HDMI 信号是否正常
v4l2-ctl -d /dev/video0 --get-dv-timings
```

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
| HDMI 采集 | ✅ GStreamer v4l2src io-mode=2, 1920×1080 |
| OpenCV 显示 | ✅ debug 模式可用，但 Qt/X11 QPixmap 会随时间泄漏，勿长期运行 |
| motion 检测 + 状态机循环 | ✅ scan → track → aim → throw → verify 完整跑通 |

### 2026-06-03 物种识别模型训练 + 控制器优化

| 项目 | 结果 |
|------|------|
| 精灵分类方案 | ✅ 从 3 类(normal/shiny/corrupted) 升级为 8 物种识别 |
| 爬虫采集 | ✅ 百度图片搜索可用（需代理），百度 CDN 缩略图可直接下载 |
| 视频帧提取 | ✅ `extract_frames.py` 支持去重/间隔采样，OBS 框选区域录制最佳 |
| 自动标注 | ✅ 运动检测 + 边缘检测自动生成 YOLO 标注框 |
| 小独角兽模型 | ✅ mAP50=0.995, mAP50-95=0.966, ONNX 11.7MB |
| PyTorch CUDA | ✅ nightly cu128 支持 RTX 5060 Ti (Blackwell sm_120) |
| 人类行为模拟 | ✅ 鼠标轨迹随机抖动、过冲回正、速度变化 |
| 长按瞄准默认 | ✅ `default_aim_mode: hold_aim` 先瞄准再丢球 |
| 视角自动旋转 | ✅ SCAN 状态连续空扫描触发全景搜索 |
| 配置文件更新 | ✅ config.yaml 适配 8 物种 + humanize + pan 参数 |

### 2026-06-03 二次瞄准 + 抛物线修正 + GStreamer 叠加显示

| 项目 | 结果 |
|------|------|
| 四步精确瞄准 | ✅ 移动到精灵 → 按住左键 → 从中心拖拽（抛物线修正）→ 2s 等待 |
| 抛物线修正 | ✅ `parabolic_factor: 0.15`，749px 距离修正 112px，826px 修正 123px |
| 状态机循环 | ✅ CAPTURE_SUCCESS → SCAN（不卡 IDLE），自动连续捕捉 |
| GStreamer textoverlay | ✅ 画面实时显示 STATE / Backend / Target / Throws / DET 坐标 |
| 类别名修正 | ✅ 单类模型 `classes: [xiao_dujiaoshou]`，不再误显示为 huzhu_quan |
| 热键反馈 | ✅ 按键时打印 `🔑 Hotkey: f3 → pause` 日志 |
| 丢球验证 | ✅ 成功/失败后自动循环继续 |
| 内存 | ✅ 156-170MB 稳定 |
| 采集帧率 | ✅ 10fps，GStreamer `max-size-buffers=2 leaky=downstream` |

### 2026-06-03 稳定性修复 + 内存优化

| 项目 | 结果 |
|------|------|
| 鼠标按键状态 bug 修复 | ✅ `move()` 现在保持当前按键状态，瞄准微调不再释放左键 |
| Auto 模式画面显示 | ✅ GStreamer 原生 `autovideosink` 窗口（tee 两路，1280×720 显示 + BGR appsink 推理）|
| Qt 内存泄漏修复 | ✅ Auto 模式完全绕过 OpenCV highgui，RSS 稳定在 ~150MB |
| 内存优化 | ✅ GC + malloc_trim 每 5 循环 + 每 15 秒定时；GStreamer 队列限制 max-buffers=2 |
| 采集帧率 | ✅ 从 30fps 降至 10fps（检测只需 ~2fps，减少内存分配压力）|
| Qt 警告消除 | ✅ `QT_LOGGING_RULES` + `G_MESSAGES_DEBUG` 环境变量抑制无害警告 |
| 错误恢复 | ✅ 状态机连续 5 次错误后自动重置；所有 NPU 推理步骤 try/except 保护 |
| RKNN NPU 推理 | ✅ `librknnrt.so` ctypes 封装，3 核 NPU 推理 ~20ms |
| 状态机完整跑通 | ✅ SCAN → TRACK → AIM → THROW → VERIFY → IDLE 全流程 OK |
| 运行内存 | ✅ 147MB 稳定，持续 15s 无增长 |

### 2026-06-03 auto/debug-auto 卡帧与检测差异修复（待实机验证）

| 项目 | 结果 |
|------|------|
| AIM 报错 | ✅ 修复 `state_machine.py` 中 `sw` 未定义，二次抛物线修正现在可正常计算 |
| appsink 过载 | 🟡 ML/appsink 分支加入 `videorate drop-only=true`，按 `capture.fps` 限速，减少 1080p 帧拷贝和内存分配压力 |
| GStreamer 重启 | 🟡 停止时释放 bus signal watch，`restart()` 增加互斥锁，避免重复重启互相踩踏 |
| auto/debug-auto 误检 | 🟡 新增 `automation.detection_filter`，自动流程过滤低置信度、极小/极大、异常宽高比 bbox；debug 原始检测不变 |
| debug-auto 显示误导 | ✅ `tracker.get_active_detections()` 不再显示已丢失帧的追踪器，避免旧伪框长时间停留 |
| HID 提示 | ✅ auto 模式不再固定检查 `/dev/hidg0`，改为检查任意 `/dev/hidg*`，鼠标节点仍由 `controller.py` 自动探测 |
| 验证 | ✅ Python 语法编译通过；硬件长时间运行尚待测试 |

### 2026-06-03 移动目标跟踪 + 头部瞄准调参（待实机验证）

| 项目 | 结果 |
|------|------|
| 漏检 | 🟡 `detector.conf_threshold` 与 `automation.detection_filter.min_confidence` 从 0.15 降到 0.12 |
| 瞄准落点 | 🟡 新增 `target_y_ratio: 0.30`，自动流程用 bbox 上部作为命中点；无 bbox 时按追踪中心上移 60px |
| 移动跟踪 | 🟡 AIM 阶段用“卡尔曼平滑中心 + bbox 头部偏移”，避免直接瞄准身体中心/脚部 |
| 鼠标响应 | 🟡 `move_steps` 20→12、`step_delay` 2ms→1ms、`hold_time` 2000ms→1500ms |
| PID 速度 | 🟡 `pid_kP` 0.6→0.85、`pid_max_step` 30→50、`pid_step_wait` 0.03→0.02、`pid_max_iters` 50→70 |
| NPU 压力回退 | ✅ `charge_update_interval` 保持 0.20s，`pid_detect_interval` 保持 3，避免每轮 PID 都触发 RKNN 推理 |
| 首帧卡死保护 | ✅ auto/debug-auto 启动前必须确认帧 ID 持续增长；restart 后也必须产出新鲜帧，否则判定重启失败 |
| 验证 | ✅ Python 语法编译通过；命中率需实机观察后继续微调 |

### 2026-06-03 慢速固定方向搜索（待实机验证）

| 项目 | 结果 |
|------|------|
| 扫描方式 | ✅ 不再执行右扫→左扫→回中间的全景扫描，改成固定方向小步扫描 |
| 旋转速度 | ✅ `controller.pan.amount` 200→60，`automation.pan.step_pixels` 45，`step_delay` 0.35s |
| 左右晃动 | ✅ `alternate: false`，底层 pan 移除反向微调，只做单方向慢速移动 |
| 发现即停 | ✅ 每个 pan step 后立即检测；识别到精灵后停止转动并进入 TRACK/AIM |
| 重启失败处理 | ✅ debug-auto 中 HDMI restart 若无法产出新鲜帧，会停止 FSM 和显示循环，避免继续空转刷屏 |

### 2026-06-03 帧率同步 + 先蓄力后瞄准（待实机验证）

| 项目 | 结果 |
|------|------|
| PC 输出刷新率 | 30Hz 更稳定；Orange Pi 侧采集目标从 10fps 提到 15fps |
| 扫描响应 | `automation.scan_interval` 0.5s→0.35s |
| 固定方向搜索速度 | `controller.pan.amount` 60→80，`automation.pan.step_pixels` 45→60，`step_delay` 0.35s→0.25s |
| AIM 顺序 | ✅ 改为先按住左键进入蓄力/瞄准状态，等待 `enter_delay` 后再通过 PID 旋转/拖拽对准 |
| 偏差修复 | ✅ 不再 hold 前先移动到精灵，避免游戏进入长按状态时准星重置造成落点偏移 |

### 2026-06-03 20fps + 长按搜索 + 出球确认（待实机验证）

| 项目 | 结果 |
|------|------|
| 采集帧率 | 🟡 `capture.fps` 15→20；若再次卡帧可回退到 15 |
| 固定方向搜索速度 | 🟡 `controller.pan.amount` 80→100，`automation.pan.step_pixels` 60→80，`step_delay` 0.25s→0.20s |
| 长按搜索 | 🟡 `hold_during_search: true`，右向搜索时保持左键长按；找到目标后直接进入 AIM，不释放左键 |
| 出球前确认 | ✅ PID 对齐后必须连续 2 次确认目标仍在准星附近，误差阈值 14px |
| 防误丢 | ✅ 确认失败时保持 AIM 状态继续修正，不进入 THROW，不主动松手 |

### 2026-06-03 30fps + 更快搜索 + 3 次确认（待实机验证）

| 项目 | 结果 |
|------|------|
| 采集帧率 | 🟡 `capture.fps` 20→30，与 PC 端 30Hz 输出对齐 |
| 固定方向搜索速度 | 🟡 `automation.pan.step_delay` 0.20s→0.15s，`step_pixels` 保持 80 |
| 出球前确认 | 🟡 `pre_throw_confirmations` 2→3，提高出球稳定性 |

### 2026-06-03 瞄准确认闭环 + 高频丢球（待实机验证）

| 项目 | 结果 |
|------|------|
| 确认时机 | ✅ 出球前确认在 PID 对齐后执行；确认通过后才进入 THROW 松手 |
| 确认闭环 | ✅ 确认阶段若发现偏移，会立即微调并把 cross 偏移带回后续复瞄，不再重置 |
| 瞄准灵敏度 | 🟡 `pid_kP` 0.85→0.95，`pid_max_step` 50→60，`pid_step_wait` 0.02→0.015，`pid_align_threshold` 10→8，`pid_detect_interval` 3→2 |
| 确认严格度 | 🟡 连续 3 次确认，阈值 14px→10px，最多检查 8 次，失败后最多追加 2 轮复瞄 |
| 丢球频率 | 🟡 `verify_wait` 3.0s→1.8s，THROW 后固定等待 0.3s→0.12s |

### 2026-06-03 多目标场景关闭出球确认

| 项目 | 结果 |
|------|------|
| 出球前确认 | ❌ 已关闭，`pre_throw_confirmations: 0` |
| 原因 | 多目标场景下确认阶段容易被其他目标抢走 tracker/target，导致反复误修正 |
| 当前策略 | PID 对齐后立即进入 THROW，减少多目标干扰窗口 |

### 2026-06-03 PID 抖动收敛 + 背景过滤收紧

| 项目 | 结果 |
|------|------|
| PID 抖动 | ✅ `pid_kP` 0.95→0.70，`pid_max_step` 60→35，`pid_align_threshold` 8→12，`pid_detect_interval` 2→3 |
| 小步死区 | ✅ 新增 `pid_min_step: 2`，小于 2px 的修正忽略，减少准星来回抖 |
| 瞄准点平滑 | ✅ 新增 `aim_smoothing: 0.65`，对 bbox 头部瞄准点做 EMA 平滑 |
| 背景干扰 | ✅ `conf_threshold/min_confidence` 0.12→0.16，过滤框面积和宽高比收紧 |

### 2026-06-04 移动目标方案分析

| 项目 | 结论 |
|------|------|
| 移动中丢目标 | 不是单纯 YOLO 或训练数据问题，而是单帧检测缺少连续性 |
| 后续方向 | 检测、跟踪、预测、出球门控分层处理 |
| 运动检测 | 可作为移动候选和短暂丢失辅助，但不替代 YOLO 类别识别 |
| World model | 不作为第一优先级；固定场景更适合用 ROI、屏蔽区、尺寸/高度先验 |
| 控制策略 | PID 只负责移动准星，是否松手应由稳定性和轨迹预测判断 |

### 2026-06-03 debug-auto 卡帧排查（历史记录）

| 项目 | 结果 |
|------|------|
| 帧队列 2→4 | ❌ 无效 |
| 显示线程 `get_frame_nonblock()` | ❌ 无效 |
| PID 检测间隔 throttling（3次1检）| ❌ 无效 |
| `_on_new_sample` try/except 防崩 | ❌ 无效 |
| 总线 EOS/ERROR + `is_stale` + auto-restart | ❌ 无效（管道未报任何错误） |
| `v4l2src io-mode=2`（MMAP）| ❌ 无效 |
| 诊断结论 | `get_frame()` 持续返回同一帧，`new-sample` 信号停止触发，管道名义上仍在 PLAYING |
| 疑似根因 | USB HID 写入阻塞 / HDMI 信号变化 / CMA 耗尽 / GLib 线程静默退出（见已知问题章节） |

### 2026-06-03 画面卡住 + 丢球精度修复

| 项目 | 结果 |
|------|------|
| 画面 10 秒后卡住 | ✅ 根因：`capture.py` GStreamer 管道 `num-buffers=100`，移除后管道持续运行 |
| GC 定时回收不执行 | ✅ 根因：`_gc_loop` while 循环体为空，`_trim_memory()` 从未被调用 |
| 抛物线修正线性→二次 | ✅ `offset = dist² × pf / sw`（二次公式），`parabolic_factor` 0.15→0.4 |
| AIM 目标丢失浪费丢球 | ✅ 目标丢失后释放按键 → `THROW_DONE` → VERIFY（不再白丢） |
| 误检过多 | ✅ `conf_threshold` 从 0.05 提高到 0.15 |
| PID 参数优化 | ✅ `pid_step_wait` 0.05→0.03，`pid_max_iters` 30→50，`pid_align_threshold` 8→10 |

### 2026-06-03 V4 昼夜模型 + 部署就绪

| 项目 | 结果 |
|------|------|
| 训练数据 | 2段视频(58帧) + 夜间10张 + 白天9张 → 1617张增强 |
| 夜间检测 | ✅ 10/10，置信度 0.86-0.97 |
| 白天检测 | ✅ 9/10，置信度 0.43-0.83 |
| 综合检出率 | **95% (19/20)** |
| ONNX 模型 | models/sprite_detector.onnx (11.7MB) |
| 部署 | 拷贝到开发板 `models/` 目录即可使用 |

### PC 训练工作流

> 训练在 Windows PC (5060 Ti 16GB) 上完成，ONNX 模型部署到 Orange Pi。

```bash
# 完整流程：录视频 → 提取帧 → 自动标注 → 训练
# 详见上方 "2. 放置模型文件" 章节

# 1. 录视频 (OBS 框选精灵区域)
# 2. 提取帧
python scripts/extract_frames.py dataset/video/xxx.mp4 --sprite xiao_dujiaoshou --fps 2
# 3. 自动标注
python scripts/auto_label.py dataset/raw_xiao_dujiaoshou/ --class-id 6
# 4. 增量训练
python scripts/incremental_train.py --new-data dataset/labeled_xiao_dujiaoshou/ --device cuda
# 5. 部署
scp models/sprite_detector.onnx orangepi@<ip>:/home/orangepi/Desktop/rock/models/
```

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
