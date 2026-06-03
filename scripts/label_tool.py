#!/usr/bin/env python3
"""
图像标注工具 — 在截图上画边界框，生成 YOLO 格式标注文件。

用法:
    python3 scripts/label_tool.py dataset/raw/ [--classes normal,shiny,corrupted]

操作:
    鼠标拖拽  — 画矩形框
    1-8        — 切换当前类别 (护主犬/伊贝儿/恶魔叮/菊花梨/公平鸽/灵狐/小独角兽/小夜)
    D          — 删除最后画的框
    ENTER      — 保存标注并加载下一张图
    ESC        — 跳过当前图片 (不保存)
    Q          — 退出程序

输出:
    标注文件保存为同名的 .txt 文件 (YOLO 格式: class_id cx cy w h)
    标注后的图片和标签会被移动到 dataset/labeled/ 目录
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# 默认类别
# ============================================================
# S2 赛季精灵物种分类 (可出异色的8只常驻精灵)
DEFAULT_CLASSES = [
    "huzhu_quan",         # 0: 护主犬 (音速犬)
    "yibei_er",           # 1: 伊贝儿
    "emo_ding",           # 2: 恶魔叮
    "juhua_li",           # 3: 菊花梨
    "gongping_ge",        # 4: 公平鸽
    "ling_hu",            # 5: 灵狐
    "xiao_dujiaoshou",    # 6: 小独角兽
    "xiaoye_yifu",        # 7: 小夜/朔夜伊芙
]

CLASS_COLORS = {
    0: (0, 255, 0),      # 护主犬 → 绿色
    1: (255, 128, 0),    # 伊贝儿 → 橙色
    2: (255, 0, 255),    # 恶魔叮 → 紫色
    3: (0, 255, 255),    # 菊花梨 → 黄色
    4: (255, 0, 0),      # 公平鸽 → 蓝色
    5: (0, 128, 255),    # 灵狐 → 天蓝
    6: (128, 0, 255),    # 小独角兽 → 紫罗兰
    7: (255, 255, 0),    # 小夜 → 青色
}


class LabelTool:
    def __init__(self, image_dir: str, classes: list, output_dir: str = None):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else self.image_dir.parent / "labeled"
        self.classes = classes
        self.current_class = 0
        self.boxes = []          # [(x1, y1, x2, y2, class_id), ...]
        self.drawing = False
        self.start_pt = None
        self.current_pt = None
        self.image_files = []
        self.current_idx = 0
        self.frame = None
        self.display = None
        self.window_name = "Label Tool"

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)

        # 加载图片列表
        self._load_image_list()

    def _load_image_list(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in exts
        ])
        if not self.image_files:
            print(f"[ERROR] 在 {self.image_dir} 中未找到图片文件")
            sys.exit(1)
        print(f"找到 {len(self.image_files)} 张图片")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.current_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_pt and self.current_pt:
                x1, y1 = self.start_pt
                x2, y2 = self.current_pt
                # 确保 x1<x2, y1<y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                # 过滤太小的框
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.boxes.append((x1, y1, x2, y2, self.current_class))
                    print(f"  框 #{len(self.boxes)}: [{self.classes[self.current_class]}] "
                          f"({x1},{y1})-({x2},{y2})")

    def _draw_boxes(self):
        """在显示图像上绘制所有标注框。"""
        self.display = self.frame.copy()
        h, w = self.display.shape[:2]

        # 绘制已有框
        for i, (x1, y1, x2, y2, cls_id) in enumerate(self.boxes):
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
            label = f"{i}:{self.classes[cls_id]}"
            cv2.putText(self.display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制正在画的框
        if self.drawing and self.start_pt and self.current_pt:
            color = CLASS_COLORS.get(self.current_class, (255, 255, 255))
            cv2.rectangle(self.display, self.start_pt, self.current_pt, color, 1)

        # 状态栏
        status = (f"Image: {self.current_idx + 1}/{len(self.image_files)}  |  "
                  f"Class: [{self.current_class}] {self.classes[self.current_class]}  |  "
                  f"Boxes: {len(self.boxes)}  |  "
                  f"1-8=class  D=del  Enter=save  Esc=skip  Q=quit")
        cv2.putText(self.display, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 类别提示
        y_off = 30
        for i, (name, color) in enumerate(
            [(c, CLASS_COLORS[i]) for i, c in enumerate(self.classes)]
        ):
            marker = ">>>" if i == self.current_class else "   "
            text = f"{marker} [{name}]"
            cv2.putText(self.display, text, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_off += 25

    def _save_current(self):
        """保存当前图片的标注。"""
        if not self.boxes:
            print("  没有标注框，跳过保存")
            return False

        img_file = self.image_files[self.current_idx]
        h, w = self.frame.shape[:2]

        # 复制图片到 labeled 目录
        dst_img = self.output_dir / "images" / img_file.name
        cv2.imwrite(str(dst_img), self.frame)

        # 生成 YOLO 格式标注
        label_lines = []
        for x1, y1, x2, y2, cls_id in self.boxes:
            # YOLO 格式: class_id cx cy w h (归一化)
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")

        dst_label = self.output_dir / "labels" / f"{img_file.stem}.txt"
        dst_label.write_text("\n".join(label_lines))

        print(f"  ✅ 已保存: {dst_img.name} + {len(label_lines)} 个标注框")
        return True

    def load_image(self, idx):
        """加载指定索引的图片。"""
        if 0 <= idx < len(self.image_files):
            path = self.image_files[idx]
            self.frame = cv2.imread(str(path))
            if self.frame is None:
                print(f"[WARN] 无法读取: {path}")
                return False
            self.boxes = []
            self._draw_boxes()
            cv2.imshow(self.window_name, self.display)
            return True
        return False

    def run(self):
        """主循环。"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        if not self.load_image(self.current_idx):
            print("无法加载图片")
            return

        print(f"\n标注工具已就绪 | {len(self.image_files)} 张图片待标注")
        print(f"类别: {self.classes}")
        print(f"输出: {self.output_dir.resolve()}")

        while True:
            self._draw_boxes()
            cv2.imshow(self.window_name, self.display)

            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("退出程序")
                break

            elif ord('1') <= key <= ord('8'):
                cls_idx = key - ord('1')
                if cls_idx < len(self.classes):
                    self.current_class = cls_idx
                    print(f"当前类别: [{cls_idx}] {self.classes[cls_idx]}")

            elif key == ord('d'):
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"  已删除框 #{len(self.boxes)}: {removed}")
                else:
                    print("  没有框可删除")

            elif key == 13:  # Enter
                self._save_current()
                self.current_idx += 1
                if self.current_idx >= len(self.image_files):
                    print("所有图片已标注完成！")
                    break
                if not self.load_image(self.current_idx):
                    break

            elif key == 27:  # Esc — 跳过
                print(f"  跳过: {self.image_files[self.current_idx].name}")
                self.current_idx += 1
                if self.current_idx >= len(self.image_files):
                    print("所有图片已处理")
                    break
                if not self.load_image(self.current_idx):
                    break

        cv2.destroyAllWindows()
        print(f"\n标注完成 | 输出目录: {self.output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 图像标注工具 — 在截图上画框生成训练数据")
    parser.add_argument("image_dir", help="截图目录路径")
    parser.add_argument("--classes", default="huzhu_quan,yibei_er,emo_ding,juhua_li,"
                        "gongping_ge,ling_hu,xiao_dujiaoshou,xiaoye_yifu",
                        help="类别名，逗号分隔")
    parser.add_argument("--output", default=None,
                        help="标注输出目录 (默认: {image_dir}/../labeled)")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    tool = LabelTool(args.image_dir, classes, args.output)
    tool.run()


if __name__ == "__main__":
    main()
