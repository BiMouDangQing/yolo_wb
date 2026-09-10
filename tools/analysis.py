"""
数据集分析命令行工具：统计标注情况、剔除未标注图片、生成分析 CSV。

与前端模块 `QT/modules/analysis.py` 对应，逻辑保持一致。

用法:
    python tools/analysis.py -i ./images -l ./labels -o ./analysis.csv
    python tools/analysis.py -i ./images -l ./labels -u ./unlabeled --remove --move
"""

import argparse
import csv
import shutil
from pathlib import Path

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def parse_label_file(path):
    """解析一个 YOLO 标签文件，返回 class_id 列表。"""
    classes = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
    except OSError:
        pass
    return classes


def resolve_class_names(labels_dir, user_names):
    if user_names:
        return [n.strip() for n in user_names.split(",") if n.strip()]
    classes_txt = labels_dir / "classes.txt"
    if classes_txt.is_file():
        return [
            line.strip()
            for line in classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    return []


def name_of(names, class_id):
    return names[class_id] if 0 <= class_id < len(names) else str(class_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集分析：统计标注、剔除未标注、生成 CSV")
    parser.add_argument("--input", "-i", required=True, help="images 目录")
    parser.add_argument("--labels", "-l", default="", help="labels 目录（留空=与图片同目录）")
    parser.add_argument("--unlabeled", "-u", default="", help="未标注图片输出目录（配合 --remove）")
    parser.add_argument("--output", "-o", default="", help="CSV 输出路径（留空= images 下 dataset_analysis.csv）")
    parser.add_argument("--remove", action="store_true", help="剔除未标注图片")
    parser.add_argument("--move", action="store_true", help="移动而非复制（配合 --remove）")
    parser.add_argument("--classes", "-c", default="", help="类别名，逗号分隔，如 durian,background")
    args = parser.parse_args()

    images = Path(args.input).expanduser().resolve()
    if not images.is_dir():
        raise SystemExit(f"images 目录不存在: {images}")

    files = sorted(
        p for p in images.rglob("*")
        if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
    )
    if not files:
        raise SystemExit("images 目录中没有找到图片。")

    labels_dir = Path(args.labels).expanduser().resolve() if args.labels else None
    names = resolve_class_names(labels_dir if labels_dir else images, args.classes)

    class_counts = {}
    total_boxes = 0
    annotated = 0
    unlabeled_list = []
    details = []

    for img in files:
        rel = img.relative_to(images)
        label_path = (labels_dir / rel.with_suffix(".txt")) if labels_dir else img.with_suffix(".txt")
        boxes = parse_label_file(label_path) if label_path.is_file() else []
        if boxes:
            annotated += 1
            total_boxes += len(boxes)
            for cls in boxes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            details.append((rel.as_posix(), "是", len(boxes)))
        else:
            unlabeled_list.append(img)
            details.append((rel.as_posix(), "否", 0))

    unlabeled = len(unlabeled_list)

    removed = 0
    if args.remove and unlabeled_list:
        unlabeled_dir = Path(args.unlabeled).expanduser().resolve() if args.unlabeled else images.parent / "unlabeled"
        unlabeled_dir.mkdir(parents=True, exist_ok=True)
        for img in unlabeled_list:
            dst = unlabeled_dir / img.name
            try:
                if args.move:
                    shutil.move(str(img), str(dst))
                else:
                    shutil.copy2(img, dst)
                removed += 1
            except OSError as exc:
                print(f"[失败] {img.name}: {exc}")
        print(f"未标注图片 {unlabeled} 张，已处理 {removed} 张到: {unlabeled_dir}")

    # 生成 CSV
    if args.output:
        csv_path = Path(args.output).expanduser().resolve()
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path / "dataset_analysis.csv"
    else:
        csv_path = images / "dataset_analysis.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值"])
        writer.writerow(["图片总数", len(files)])
        writer.writerow(["已标注", annotated])
        writer.writerow(["未标注", unlabeled])
        writer.writerow(["总标注框", total_boxes])
        writer.writerow([])
        writer.writerow(["class_id", "类别", "框数量"])
        for cls in sorted(class_counts):
            writer.writerow([cls, name_of(names, cls), class_counts[cls]])
        writer.writerow([])
        writer.writerow(["图片", "是否标注", "框数量"])
        for row in details:
            writer.writerow(list(row))

    print(f"\n完成: 共 {len(files)} 张，已标注 {annotated} 张，未标注 {unlabeled} 张，总框数 {total_boxes}。")
    print(f"CSV 已生成: {csv_path}")


if __name__ == "__main__":
    main()
