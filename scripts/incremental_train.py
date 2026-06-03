#!/usr/bin/env python3
"""
增量训练管理工具 — 支持按物种分批添加训练数据。

场景:
  用户按精灵物种分批采集数据，每批只包含1-2种精灵。
  本工具管理数据的累积合并，支持增量训练。

用法:
    # 第一批: 只采集了小独角兽
    python scripts/incremental_train.py --new-data dataset/labeled_xiao_dujiaoshou/

    # 第二批: 追加护主犬数据
    python scripts/incremental_train.py --new-data dataset/labeled_huzhu_quan/

    # 第三批: 追加伊贝儿数据
    python scripts/incremental_train.py --new-data dataset/labeled_yibei_er/

    # ...直到所有8种精灵全部添加

工作流程:
  1. 将新数据合并到累积数据集 dataset/cumulative/
  2. 自动运行数据增强
  3. 重新训练模型 (使用当前已有的所有类别)
  4. 导出 ONNX 模型

目录结构:
  dataset/
    ├── labeled_xiao_dujiaoshou/    # 第一批: 小独角兽标注数据
    │   ├── images/
    │   └── labels/
    ├── labeled_huzhu_quan/         # 第二批: 护主犬标注数据
    │   ├── images/
    │   └── labels/
    ├── cumulative/                  # 累积合并后的数据
    │   ├── images/
    │   └── labels/
    └── cumulative_augmented/       # 增强后的训练数据
        ├── images/
        └── labels/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 完整的8个物种类别定义
ALL_CLASSES = [
    "huzhu_quan",         # 0: 护主犬
    "yibei_er",           # 1: 伊贝儿
    "emo_ding",           # 2: 恶魔叮
    "juhua_li",           # 3: 菊花梨
    "gongping_ge",        # 4: 公平鸽
    "ling_hu",            # 5: 灵狐
    "xiao_dujiaoshou",    # 6: 小独角兽
    "xiaoye_yifu",        # 7: 小夜/朔夜伊芙
]

CLASS_NAMES_CN = {
    "huzhu_quan": "护主犬",
    "yibei_er": "伊贝儿",
    "emo_ding": "恶魔叮",
    "juhua_li": "菊花梨",
    "gongping_ge": "公平鸽",
    "ling_hu": "灵狐",
    "xiao_dujiaoshou": "小独角兽",
    "xiaoye_yifu": "小夜/朔夜伊芙",
}


def merge_datasets(new_data_dir: Path, cumulative_dir: Path):
    """
    将新标注数据合并到累积数据集。

    策略:
      - 如果 cumulative/ 不存在，直接复制新数据
      - 如果已存在，追加新文件 (不覆盖已有文件)
      - 自动检测新数据包含哪些类别
    """
    new_img_dir = new_data_dir / "images"
    new_lbl_dir = new_data_dir / "labels"

    if not new_img_dir.exists() or not new_lbl_dir.exists():
        raise FileNotFoundError(
            f"新数据目录不完整: images={new_img_dir.exists()}, "
            f"labels={new_lbl_dir.exists()}"
        )

    cum_img_dir = cumulative_dir / "images"
    cum_lbl_dir = cumulative_dir / "labels"
    cum_img_dir.mkdir(parents=True, exist_ok=True)
    cum_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 检测新数据中的类别
    active_classes = set()
    for lbl_file in new_lbl_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(ALL_CLASSES):
                        active_classes.add(ALL_CLASSES[cls_id])

    # 复制文件
    existing = set(f.stem for f in cum_img_dir.glob("*"))
    copied = 0
    skipped = 0

    for img_path in sorted(new_img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        lbl_path = new_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"  [SKIP] 无标注: {img_path.name}")
            continue

        # 避免重复: 使用新数据的目录名作为前缀
        prefix = new_data_dir.name.replace("labeled_", "")
        dest_name = f"{prefix}_{img_path.stem}"

        if dest_name in existing:
            skipped += 1
            continue

        shutil.copy2(img_path, cum_img_dir / f"{dest_name}{img_path.suffix}")
        shutil.copy2(lbl_path, cum_lbl_dir / f"{dest_name}.txt")
        copied += 1
        existing.add(dest_name)

    print(f"  合并完成: 新增 {copied} 张, 跳过 {skipped} 张 (重复)")
    print(f"  累积数据集总计: {len(existing)} 张")

    return active_classes


def check_cumulative_classes(cumulative_dir: Path) -> list:
    """
    扫描累积数据集，返回当前已有哪些类别。
    用于生成只包含已有类别的 data.yaml。
    """
    active_ids = set()
    lbl_dir = cumulative_dir / "labels"
    if not lbl_dir.exists():
        return []

    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    active_ids.add(cls_id)

    return [ALL_CLASSES[i] for i in sorted(active_ids) if i < len(ALL_CLASSES)]


def main():
    parser = argparse.ArgumentParser(
        description="增量训练管理 — 按物种分批添加训练数据"
    )
    parser.add_argument("--new-data", required=True,
                        help="新标注数据目录 (如 dataset/labeled_xiao_dujiaoshou/)")
    parser.add_argument("--cumulative-dir", default="dataset/cumulative",
                        help="累积数据集目录 (默认 dataset/cumulative/)")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="是否运行数据增强 (默认: 是)")
    parser.add_argument("--augment-factor", type=int, default=10,
                        help="增强倍数 (默认 10)")
    parser.add_argument("--train", action="store_true", default=True,
                        help="是否自动训练 (默认: 是)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认 100)")
    parser.add_argument("--device", default="cpu",
                        help="训练设备 (cpu/cuda)")
    parser.add_argument("--no-train", action="store_true",
                        help="跳过训练，仅合并数据")
    args = parser.parse_args()

    new_data_dir = Path(args.new_data)
    cumulative_dir = Path(args.cumulative_dir)

    print("=" * 60)
    print("增量训练流水线")
    print("=" * 60)

    # Step 1: 合并数据
    print("\n📦 Step 1: 合并数据到累积数据集")
    active_classes = merge_datasets(new_data_dir, cumulative_dir)

    if active_classes:
        names_cn = [f"{c}({CLASS_NAMES_CN.get(c, c)})" for c in sorted(active_classes)]
        print(f"  当前活跃类别: {', '.join(names_cn)}")

    # 检测所有已有类别
    all_active = check_cumulative_classes(cumulative_dir)
    print(f"  累积数据集总类别数: {len(all_active)}/8")
    for i, cls in enumerate(all_active):
        cn = CLASS_NAMES_CN.get(cls, cls)
        print(f"    {i}: {cn} ({cls})")

    if args.no_train:
        print("\n⏸️  跳过训练 (--no-train)")
        print(f"累积数据集位于: {cumulative_dir.resolve()}")
        return

    # Step 2: 数据增强
    print("\n🔧 Step 2: 数据增强")
    from scripts.augment_dataset import DataAugmentor

    aug_dir = cumulative_dir.parent / "cumulative_augmented"
    if aug_dir.exists():
        shutil.rmtree(aug_dir)

    augmentor = DataAugmentor(seed=42)
    augmentor.augment_dataset(
        labeled_dir=cumulative_dir,
        output_dir=aug_dir,
        factor=args.augment_factor,
        intensity="medium",
    )

    # Step 3: 准备 YOLO 训练目录
    print("\n📋 Step 3: 准备训练数据")

    import random
    random.seed(42)

    # 划分 train/val
    images = sorted(list((aug_dir / "images").glob("*.png")) +
                    list((aug_dir / "images").glob("*.jpg")))

    paired = []
    for img_path in images:
        lbl_path = aug_dir / "labels" / f"{img_path.stem}.txt"
        if lbl_path.exists():
            paired.append((img_path, lbl_path))

    random.shuffle(paired)
    n_val = max(1, int(len(paired) * 0.2))
    val_pairs = paired[:n_val]
    train_pairs = paired[n_val:]

    yolo_dir = cumulative_dir.parent / "yolo_format"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, yolo_dir / "images" / split / img_path.name)
            shutil.copy2(lbl_path, yolo_dir / "labels" / split / lbl_path.name)

    print(f"  训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}")

    # 创建 data.yaml (只包含已有类别)
    import yaml
    data_yaml_path = yolo_dir / "data.yaml"
    active_classes_sorted = check_cumulative_classes(cumulative_dir)
    yaml_content = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(active_classes_sorted)},
        "nc": len(active_classes_sorted),
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml 已创建 ({len(active_classes_sorted)} 个类别)")

    # Step 4: 训练
    print(f"\n🚀 Step 4: 训练 YOLOv8 ({args.epochs} epochs, device={args.device})")
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.train(
            data=str(data_yaml_path),
            epochs=args.epochs,
            imgsz=640,
            batch=8,
            device=args.device,
            workers=2,
            patience=20,
            save=True,
            project="runs/train",
            name="sprite_detector_incremental",
            exist_ok=True,
            verbose=True,
        )

        # 找到 best.pt
        best_pt = Path("runs/train/sprite_detector_incremental/weights/best.pt")

        # 复制到 models/
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        shutil.copy2(best_pt, models_dir / "sprite_detector.pt")
        print(f"  模型保存至: models/sprite_detector.pt")

        # Step 5: 导出 ONNX
        print("\n📤 Step 5: 导出 ONNX")
        model = YOLO(str(best_pt))
        model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
        )

        onnx_file = best_pt.with_suffix(".onnx")
        if onnx_file.exists():
            shutil.copy2(onnx_file, models_dir / "sprite_detector.onnx")
            print(f"  ONNX 模型导出至: models/sprite_detector.onnx")
        else:
            print("  ⚠️ ONNX 导出可能失败，请检查")

    except ImportError:
        print("  ❌ 需要安装 ultralytics: pip install ultralytics")
        print(f"  数据已准备就绪: {yolo_dir}")
        print(f"  请手动运行: python scripts/train_model.py --data {yolo_dir}")

    # 完成
    print("\n" + "=" * 60)
    print("✅ 增量训练完成!")
    print(f"  累积数据集: {cumulative_dir.resolve()}")
    print(f"  当前类别: {len(active_classes_sorted)}/8")
    print(f"  下一步: 采集更多精灵数据，再次运行本脚本追加")
    print("=" * 60)


if __name__ == "__main__":
    main()
