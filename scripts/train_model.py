#!/usr/bin/env python3
"""
YOLOv8 精灵检测模型训练 + ONNX 导出脚本。

用法:
    # 完整流程: 训练 + 导出 ONNX
    python3 scripts/train_model.py --data dataset/labeled/

    # 仅导出 (已有 .pt 权重文件)
    python3 scripts/train_model.py --export-only --weights models/sprite_detector.pt

    # 自定义参数
    python3 scripts/train_model.py --data dataset/labeled/ --model yolov8n.pt --epochs 100 --imgsz 640

数据集目录结构:
    dataset/labeled/
    ├── images/          # *.jpg, *.png
    │   ├── screenshot_0001.jpg
    │   ├── screenshot_0002.jpg
    │   └── ...
    └── labels/          # *.txt (YOLO 格式)
        ├── screenshot_0001.txt
        ├── screenshot_0002.txt
        └── ...

输出:
    models/sprite_detector.onnx  — ONNX 模型 (CPU 推理)
    models/sprite_detector.pt    — PyTorch 权重 (可用于后续 RKNN 转换)
"""

import argparse
import sys
import os
import shutil
import random
from pathlib import Path

import yaml

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import log

# ============================================================
# 默认配置
# ============================================================

# S2 赛季精灵物种分类 (可出异色的8只常驻精灵)
# 训练时只识别普通种族的精灵外形，异色版本外形相同仅颜色不同
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

# 类别中文名映射 (用于显示)
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

DEFAULT_MODEL = "yolov8n.pt"   # YOLOv8 nano (最小最快)
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_VAL_SPLIT = 0.2        # 20% 验证集


def prepare_dataset(labeled_dir: Path, output_dir: Path,
                    val_split: float = 0.2) -> dict:
    """
    将 labeled/ 目录整理成 YOLO 训练格式。

    labeled/
      images/  →  dataset/images/train/ + dataset/images/val/
      labels/  →  dataset/labels/train/ + dataset/labels/val/

    Returns: 数据集统计信息
    """
    img_dir = labeled_dir / "images"
    lbl_dir = labeled_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(
            f"目录结构不完整: images={img_dir.exists()}, labels={lbl_dir.exists()}"
        )

    # 获取所有图片
    images = sorted(list(img_dir.glob("*.jpg")) +
                    list(img_dir.glob("*.png")) +
                    list(img_dir.glob("*.jpeg")))

    # 只保留有对应标注文件的图片
    paired = []
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            paired.append((img_path, lbl_path))
        else:
            log.warning(f"跳过 (无标注): {img_path.name}")

    if len(paired) == 0:
        raise ValueError("没有找到带标注的图片对！请先运行 label_tool.py 标注。")

    # 打乱并划分
    random.shuffle(paired)
    n_val = max(1, int(len(paired) * val_split))
    val_pairs = paired[:n_val]
    train_pairs = paired[n_val:]

    # 清理并创建输出目录
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (output_dir / sub / split).mkdir(parents=True, exist_ok=True)

    # 复制文件
    def copy_pairs(pairs, split):
        for img_path, lbl_path in pairs:
            ext = img_path.suffix
            shutil.copy2(img_path, output_dir / "images" / split / f"{img_path.stem}{ext}")
            shutil.copy2(lbl_path, output_dir / "labels" / split / lbl_path.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    stats = {"train": len(train_pairs), "val": len(val_pairs), "total": len(paired)}
    log.info(f"数据集划分: 训练 {stats['train']} + 验证 {stats['val']} = 总计 {stats['total']}")
    return stats


def create_data_yaml(output_dir: Path, classes: list) -> Path:
    """创建 data.yaml 配置文件。"""
    yaml_path = output_dir / "data.yaml"
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    log.info(f"已创建 data.yaml: {yaml_path}")
    return yaml_path


def train_model(data_yaml: Path, model_name: str, epochs: int,
                imgsz: int, batch: int, device: str = "cpu") -> Path:
    """
    使用 ultralytics 训练 YOLO 模型。

    Returns: 训练好的模型路径 (best.pt)
    """
    from ultralytics import YOLO

    # 加载预训练模型 (或从头训练)
    log.info(f"加载基础模型: {model_name}")
    model = YOLO(model_name)

    log.info(f"开始训练 (epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device})")
    log.info(f"数据集: {data_yaml}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        patience=20,            # 20 epoch 无提升则早停
        save=True,
        save_period=10,
        project="runs/train",
        name="sprite_detector",
        exist_ok=True,
        verbose=True,
    )

    # 找到 best.pt
    best_pt = Path("runs/train/sprite_detector/weights/best.pt")
    if not best_pt.exists():
        # 尝试其他可能路径
        candidates = sorted(Path("runs/train/sprite_detector/weights").glob("*.pt"))
        if candidates:
            best_pt = candidates[-1]

    log.info(f"训练完成，模型保存至: {best_pt}")
    return best_pt


def export_to_onnx(pt_path: Path, output_path: Path,
                   imgsz: int = 640, opset: int = 12) -> bool:
    """
    将 PyTorch 权重导出为 ONNX 格式。
    """
    from ultralytics import YOLO

    log.info(f"加载权重: {pt_path}")
    model = YOLO(str(pt_path))

    log.info(f"导出 ONNX (imgsz={imgsz}, opset={opset})...")
    try:
        # ultralytics 内置的 export 方法
        success = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,
        )

        # 导出后文件在 pt_path 旁边，名为 best.onnx
        onnx_file = pt_path.with_suffix(".onnx")
        if onnx_file.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_file, output_path)
            log.info(f"ONNX 模型已导出: {output_path}")
            return True
        else:
            log.error(f"ONNX 导出失败：未找到输出文件 {onnx_file}")
            return False

    except Exception as e:
        log.error(f"ONNX 导出异常: {e}")

        # 备选方案：手动导出
        log.info("尝试手动导出...")
        try:
            import torch
            model.model.eval()
            dummy_input = torch.randn(1, 3, imgsz, imgsz)

            torch.onnx.export(
                model.model,
                dummy_input,
                str(output_path),
                opset_version=opset,
                input_names=["images"],
                output_names=["output0"],
                dynamic_axes=None,
            )
            log.info(f"ONNX 模型已导出 (手动): {output_path}")
            return True
        except Exception as e2:
            log.error(f"手动导出也失败: {e2}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 精灵检测模型训练 + ONNX 导出")

    # 模式选择
    parser.add_argument("--data", default=None,
                        help="标注数据集目录 (如 dataset/labeled/)")
    parser.add_argument("--export-only", action="store_true",
                        help="仅导出 ONNX (需要 --weights)")
    parser.add_argument("--weights", default=None,
                        help="已有 .pt 权重文件路径")

    # 训练参数
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"基础模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"训练轮数 (默认: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"图片尺寸 (默认: {DEFAULT_IMGSZ})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"批次大小 (默认: {DEFAULT_BATCH})")
    parser.add_argument("--device", default="cpu",
                        help="训练设备 (默认: cpu, 可改为 cuda)")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                        help=f"验证集比例 (默认: {DEFAULT_VAL_SPLIT})")

    # 输出
    parser.add_argument("--output-pt", default="models/sprite_detector.pt",
                        help="PyTorch 模型输出路径")
    parser.add_argument("--output-onnx", default="models/sprite_detector.onnx",
                        help="ONNX 模型输出路径")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    classes = DEFAULT_CLASSES

    if args.export_only:
        # --- 仅导出模式 ---
        if not args.weights:
            log.error("--export-only 需要 --weights 参数")
            sys.exit(1)

        pt_path = Path(args.weights)
        if not pt_path.exists():
            log.error(f"权重文件不存在: {pt_path}")
            sys.exit(1)

        onnx_path = Path(args.output_onnx)
        ok = export_to_onnx(pt_path, onnx_path, imgsz=args.imgsz)
        if ok:
            log.info("✅ ONNX 导出完成！")
            log.info(f"将 {onnx_path.name} 放入 models/ 即可使用")
        else:
            log.error("❌ ONNX 导出失败")
            sys.exit(1)

    else:
        # --- 完整训练模式 ---
        if not args.data:
            log.error("需要 --data 参数指定标注数据集目录")
            sys.exit(1)

        data_dir = Path(args.data)
        if not data_dir.exists():
            log.error(f"数据集目录不存在: {data_dir}")
            sys.exit(1)

        # Step 1: 准备数据集
        log.info("=" * 60)
        log.info("Step 1/4: 准备数据集")
        log.info("=" * 60)
        dataset_dir = project_root / "dataset" / "yolo_format"
        # 清理旧的划分
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        stats = prepare_dataset(data_dir, dataset_dir, args.val_split)

        if stats["total"] < 10:
            log.warning(f"⚠️ 只有 {stats['total']} 张标注图片，建议至少收集 50-100 张")

        # Step 2: 创建 data.yaml
        log.info("=" * 60)
        log.info("Step 2/4: 创建 data.yaml")
        log.info("=" * 60)
        data_yaml = create_data_yaml(dataset_dir, classes)

        # Step 3: 训练
        log.info("=" * 60)
        log.info("Step 3/4: 训练 YOLOv8 模型")
        log.info("=" * 60)
        best_pt = train_model(
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

        # 复制 .pt 到 models/
        pt_path = Path(args.output_pt)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, pt_path)
        log.info(f"PyTorch 模型已保存: {pt_path}")

        # Step 4: 导出 ONNX
        log.info("=" * 60)
        log.info("Step 4/4: 导出 ONNX 模型")
        log.info("=" * 60)
        onnx_path = Path(args.output_onnx)
        ok = export_to_onnx(best_pt, onnx_path, imgsz=args.imgsz)

        # 完成
        log.info("=" * 60)
        log.info("🎉 全部完成！")
        log.info(f"   PyTorch 模型: {pt_path}")
        if ok:
            log.info(f"   ONNX 模型:    {onnx_path}")
            log.info(f"")
            log.info(f"现在可以用 ONNX 模型运行检测了:")
            log.info(f"  python3 main.py debug")
        else:
            log.warning("   ONNX 导出未成功，需要手动导出或仅在 PC 上使用 .pt")
            log.info(f"   如需 RKNN 转换，请在 x86 PC 上运行 RKNN Toolkit")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
