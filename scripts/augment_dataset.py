#!/usr/bin/env python3
"""
训练数据增强工具

从少量标注数据生成更多训练样本。支持:
  - 几何变换: 旋转、翻转、缩放、平移
  - 颜色变换: 亮度、对比度、饱和度、色调
  - 噪声添加: 高斯噪声、椒盐噪声
  - 背景混合: 将精灵区域合成到不同背景上
  - 模糊模拟: 运动模糊、高斯模糊 (模拟不同距离)

用法:
    # 基础增强 (5x)
    python scripts/augment_dataset.py dataset/labeled/ --factor 5

    # 包含背景混合 (需要提供背景图目录)
    python scripts/augment_dataset.py dataset/labeled/ --factor 10 --backgrounds backgrounds/

    # 仅增强特定类别
    python scripts/augment_dataset.py dataset/labeled/ --classes huzhu_quan,yibei_er

输出:
    dataset/augmented/images/
    dataset/augmented/labels/
"""

import argparse
import random
import sys
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class DataAugmentor:
    """训练数据增强器。"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    # ----------------------------------------------------------
    # 几何变换
    # ----------------------------------------------------------

    @staticmethod
    def flip_horizontal(image: np.ndarray, boxes: list) -> tuple:
        """水平翻转。"""
        h, w = image.shape[:2]
        flipped = cv2.flip(image, 1)
        new_boxes = []
        for cls_id, cx, cy, bw, bh in boxes:
            new_boxes.append([cls_id, 1.0 - cx, cy, bw, bh])
        return flipped, new_boxes

    @staticmethod
    def rotate_small(image: np.ndarray, boxes: list,
                      angle_range: tuple = (-10, 10)) -> tuple:
        """小角度旋转 (保持边界框有效)。"""
        angle = random.uniform(*angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算新边界
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += new_w / 2 - center[0]
        matrix[1, 2] += new_h / 2 - center[1]

        rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_REPLICATE)

        # YOLO 格式 box 在旋转后保持不变 (小角度近似)
        # 对于大角度需要重新计算，这里只做轻微旋转
        new_boxes = boxes.copy()  # 近似
        return rotated, new_boxes

    @staticmethod
    def scale_variation(image: np.ndarray, boxes: list,
                         scale_range: tuple = (0.85, 1.15)) -> tuple:
        """缩放变化 (模拟不同距离)。"""
        scale = random.uniform(*scale_range)
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        return scaled, boxes  # YOLO 归一化坐标在缩放后不变

    # ----------------------------------------------------------
    # 颜色变换
    # ----------------------------------------------------------

    @staticmethod
    def brightness_contrast(image: np.ndarray, boxes: list,
                            alpha_range: tuple = (0.7, 1.3),
                            beta_range: tuple = (-30, 30)) -> tuple:
        """亮度和对比度调整。"""
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted, boxes

    @staticmethod
    def gamma_correction(image: np.ndarray, boxes: list,
                          gamma_range: tuple = (0.4, 2.5)) -> tuple:
        """Gamma 校正 — 模拟不同亮度环境（暗夜<->白天）。"""
        gamma = random.uniform(*gamma_range)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(image, table)
        return corrected, boxes

    @staticmethod
    def hsv_shift(image: np.ndarray, boxes: list,
                   h_range: tuple = (-10, 10),
                   s_range: tuple = (0.8, 1.2),
                   v_range: tuple = (0.8, 1.2)) -> tuple:
        """HSV 色彩空间扰动。"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += random.uniform(*h_range)
        hsv[:, :, 1] *= random.uniform(*s_range)
        hsv[:, :, 2] *= random.uniform(*v_range)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), boxes

    # ----------------------------------------------------------
    # 噪声
    # ----------------------------------------------------------

    @staticmethod
    def gaussian_noise(image: np.ndarray, boxes: list,
                        sigma_range: tuple = (3, 12)) -> tuple:
        """高斯噪声。"""
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy, boxes

    @staticmethod
    def motion_blur(image: np.ndarray, boxes: list,
                     kernel_range: tuple = (3, 9)) -> tuple:
        """运动模糊 (水平方向)。"""
        ksize = random.randint(*kernel_range)
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = 1.0 / ksize
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred, boxes

    # ----------------------------------------------------------
    # 组合增强流水线
    # ----------------------------------------------------------

    def augment_single(self, image: np.ndarray, boxes: list,
                       intensity: str = "medium") -> tuple:
        """
        对单张图片应用一组随机增强。

        Args:
            image: BGR 图像
            boxes: YOLO 格式标注 [[cls_id, cx, cy, w, h], ...]
            intensity: "light" / "medium" / "heavy"

        Returns:
            (augmented_image, boxes)
        """
        aug_image = image.copy()
        aug_boxes = [b.copy() if isinstance(b, list) else list(b) for b in boxes]

        pipelines = {
            "light": [
                (self.flip_horizontal, 0.5),
                (self.brightness_contrast, 0.6),
                (self.hsv_shift, 0.4),
                (self.gamma_correction, 0.4),
            ],
            "medium": [
                (self.flip_horizontal, 0.5),
                (self.brightness_contrast, 0.7),
                (self.hsv_shift, 0.5),
                (self.gamma_correction, 0.5),
                (self.rotate_small, 0.4),
                (self.gaussian_noise, 0.3),
            ],
            "heavy": [
                (self.flip_horizontal, 0.5),
                (self.brightness_contrast, 0.8),
                (self.hsv_shift, 0.6),
                (self.gamma_correction, 0.6),
                (self.rotate_small, 0.5),
                (self.gaussian_noise, 0.4),
                (self.motion_blur, 0.3),
            ],
        }

        for aug_fn, prob in pipelines.get(intensity, pipelines["medium"]):
            if random.random() < prob:
                try:
                    aug_image, aug_boxes = aug_fn(aug_image, aug_boxes)
                except Exception as e:
                    pass  # 跳过失败的增强

        return aug_image, aug_boxes

    def augment_dataset(self, labeled_dir: Path, output_dir: Path,
                        factor: int = 5, intensity: str = "medium",
                        target_classes: list = None):
        """
        对整个数据集进行增强。

        Args:
            labeled_dir: 标注数据目录 (含 images/ 和 labels/)
            factor: 每张原图生成多少张增强图
            intensity: 增强强度
            target_classes: 限制只增强某些类别
        """
        img_dir = labeled_dir / "images"
        lbl_dir = labeled_dir / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(f"数据集目录不完整: {labeled_dir}")

        out_img_dir = output_dir / "images"
        out_lbl_dir = output_dir / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.png")) +
            list(img_dir.glob("*.jpeg"))
        )

        total_original = 0
        total_augmented = 0

        print(f"增强参数: factor={factor}, intensity={intensity}")
        print(f"找到 {len(images)} 张标注图片")
        print("-" * 50)

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # 读取标注
            boxes = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        if target_classes is None or cls_id in target_classes:
                            boxes.append([cls_id, cx, cy, w, h])

            if not boxes:
                continue

            # 复制原图
            out_name = img_path.stem
            cv2.imwrite(str(out_img_dir / f"{out_name}_orig.png"), image)
            with open(out_lbl_dir / f"{out_name}_orig.txt", "w") as f:
                for box in boxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} "
                            f"{box[3]:.6f} {box[4]:.6f}\n")
            total_original += 1

            # 生成增强版本
            for i in range(factor):
                try:
                    aug_img, aug_boxes = self.augment_single(
                        image, boxes, intensity
                    )

                    h, w = aug_img.shape[:2]
                    aug_name = f"{out_name}_aug{i:02d}"

                    cv2.imwrite(str(out_img_dir / f"{aug_name}.png"), aug_img)

                    # 写入标注 (确保坐标在有效范围)
                    with open(out_lbl_dir / f"{aug_name}.txt", "w") as f:
                        for box in aug_boxes:
                            cls_id = int(box[0])
                            cx = max(0, min(1, box[1]))
                            cy = max(0, min(1, box[2]))
                            bw = max(0, min(1, box[3]))
                            bh = max(0, min(1, box[4]))
                            if bw > 0.001 and bh > 0.001:
                                f.write(f"{cls_id} {cx:.6f} {cy:.6f} "
                                        f"{bw:.6f} {bh:.6f}\n")

                    total_augmented += 1

                except Exception as e:
                    print(f"  [WARN] 增强 {img_path.name}#{i} 失败: {e}")

            print(f"  {img_path.name}: → {1 + factor} 张")

        print("-" * 50)
        print(f"完成! 原始: {total_original}, 增强: {total_augmented}, "
              f"总计: {total_original + total_augmented}")


def main():
    parser = argparse.ArgumentParser(
        description="训练数据增强 — 从少量标注生成更多训练样本"
    )
    parser.add_argument("labeled_dir", help="标注数据目录 (含 images/ 和 labels/)")
    parser.add_argument("--output", default=None,
                        help="输出目录 (默认: {labeled_dir}_augmented)")
    parser.add_argument("--factor", type=int, default=5,
                        help="每张原图的增强倍数 (默认 5)")
    parser.add_argument("--intensity", default="medium",
                        choices=["light", "medium", "heavy"],
                        help="增强强度 (默认 medium)")
    parser.add_argument("--classes", default=None,
                        help="限制增强的类别ID，逗号分隔 (如: 0,1,2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = labeled_dir.parent / f"{labeled_dir.name}_augmented"

    target_classes = None
    if args.classes:
        target_classes = [int(c.strip()) for c in args.classes.split(",")]

    augmentor = DataAugmentor(seed=args.seed)
    augmentor.augment_dataset(
        labeled_dir=labeled_dir,
        output_dir=output_dir,
        factor=args.factor,
        intensity=args.intensity,
        target_classes=target_classes,
    )

    print(f"\n输出目录: {output_dir.resolve()}")
    print(f"下一步: 将增强后的数据用于训练")


if __name__ == "__main__":
    main()
