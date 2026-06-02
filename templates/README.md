# 模板图片目录

此目录存放用于 UI 元素检测的模板图片 (.png 格式)。

## 模板要求

- 格式: PNG
- 尺寸: 建议 50x50 ~ 200x200 像素
- 内容: 从游戏截图中截取的目标 UI 元素

## 推荐模板

在 debug 模式下按 SPACE 键保存截图，然后从中裁剪需要的 UI 元素:

| 模板名称 | 用途 | 状态机使用阶段 |
|---------|------|--------------|
| `capture_success.png` | "捕捉成功"弹窗 | VERIFY → 判定成功 |
| `capture_fail.png` | "捕捉失败"弹窗 | VERIFY → 判定失败，重试 |
| `battle_end.png` | 战斗结束提示 | VERIFY → 判定成功 |
| `battle_ui.png` | 战斗界面特征（技能栏等）| VERIFY → 判定进入战斗 |

## 制作模板

```bash
# 1. 运行 debug 模式获取截图
python3 main.py debug
# 按 SPACE 键保存截图

# 2. 从截图中裁剪模板 (使用任意图片编辑工具)
# 只保留需要的 UI 元素，尽量去除背景
```

## 验证流程

状态机在每次丢球后会按以下顺序检查 4 种 UI 模板：

```
丢球完成 → 等待 3 秒
    ↓
Step 1: detect_ui("capture_success" / "battle_end") → ✅ 成功
    ↓ (未命中)
Step 2: detect_ui("capture_fail")                   → ❌ 失败，重试
    ↓ (未命中)
Step 3: detect_sprites() 画面中还有精灵？            → ❌ 还在，重试
    ↓ (未命中)
Step 4: detect_ui("battle_ui" / "skill_bar")        → ⚔️ 进入战斗
    ↓ (未命中)
Step 5: 默认 → ✅ 成功（无失败信号）
```

## 注意事项

- 模板匹配对**分辨率敏感**，确保游戏运行在 1920x1080
- 如果游戏窗口不是全屏，需要调整模板
- 模板匹配阈值可在 config.yaml 中调整:
  ```yaml
  detector:
    template:
      match_threshold: 0.7  # 降低可增加匹配数，升高可减少误检
  ```
