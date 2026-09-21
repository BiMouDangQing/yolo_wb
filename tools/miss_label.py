"""
漏标检测命令行工具：用模型批量推理，找出「模型检测到但标签没有」的疑似漏标。

与前端模块 `QT/modules/miss_label.py` 对应，逻辑保持一致。
依赖 ultralytics。

用法:
    python tools/miss_label.py -m ./best.pt -i ./images -l ./labels -o ./miss.csv
    python tools/miss_label.py -m ./best.pt -i ./images -l ./labels --conf 0.3 --iou 0.3
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def read_label_boxes(path):
    boxes = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    except OSError:
        pass
    return boxes


def iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="用模型找疑似漏标")
    parser.add_argument("--model", "-m", required=True, help="模型文件路径（.pt）")
    parser.add_argument("--input", "-i", required=True, help="images 目录")
    parser.add_argument("--labels", "-l", default="", help="labels 目录（留空=与图片同目录）")
    parser.add_argument("--output", "-o", default="", help="CSV 输出路径（可选）")
    parser.add_argument("--conf", "-c", type=float, default=0.3, help="置信度阈值，默认 0.3")
    parser.add_argument("--iou", "-t", type=float, default=0.3, help="IoU 匹配阈值，默认 0.3")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"缺少 ultralytics：{exc}") from exc

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    images = Path(args.input).expanduser().resolve()
    if not images.is_dir():
        raise SystemExit(f"images 目录不存在: {images}")

    files = sorted(p for p in images.rglob("*")
                   if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS)
    if not files:
        raise SystemExit("images 目录中没有找到图片。")

    labels_dir = Path(args.labels).expanduser().resolve() if args.labels else None
    names = getattr(model, "names", {})

    miss_count = 0
    miss_images = 0
    rows = []

    for img_path in files:
        rel = img_path.relative_to(images)
        label_path = (labels_dir / rel.with_suffix(".txt")) if labels_dir else img_path.with_suffix(".txt")
        label_boxes = read_label_boxes(label_path) if label_path.is_file() else []

        result = model(str(img_path), conf=args.conf, verbose=False)[0]
        h, w = result.orig_shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        misses = []
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            nb = (xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h)
            if not any(iou(nb, lb[1:]) >= args.iou for lb in label_boxes):
                misses.append((cls, conf, nb))

        if misses:
            miss_images += 1
            miss_count += len(misses)
            for cls, conf, nb in misses:
                name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                print(f"[疑似漏标] {img_path.name}: {name} conf={conf:.2f} "
                      f"框=({nb[0]:.3f},{nb[1]:.3f},{nb[2]:.3f},{nb[3]:.3f})")
                rows.append([img_path.name, name, f"{conf:.3f}",
                             f"{nb[0]:.4f}", f"{nb[1]:.4f}", f"{nb[2]:.4f}", f"{nb[3]:.4f}"])

    if args.output:
        csv_path = Path(args.output).expanduser().resolve()
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path / "miss_labels.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["图片", "类别", "置信度",
                             "归一化x1", "归一化y1", "归一化x2", "归一化y2"])
            for row in rows:
                writer.writerow(row)
        print(f"CSV 已生成: {csv_path}")

    print(f"\n完成: 共 {len(files)} 张图片，疑似漏标 {miss_count} 处，涉及 {miss_images} 张。")


if __name__ == "__main__":
    main()
