"""
把 Pascal VOC 格式的 XML 标注转换成 YOLO 训练格式的 txt，
并把没有标注的图片筛选出来复制/移动到单独文件夹。

XML 格式（LabelImg / Pascal VOC）示例:
    <annotation>
        <filename>xxx.jpg</filename>
        <size><width>1920</width><height>1080</height><depth>3</depth></size>
        <object>
            <name>类别名</name>
            <bndbox>
                <xmin>100</xmin><ymin>200</ymin>
                <xmax>300</xmax><ymax>400</ymax>
            </bndbox>
        </object>
    </annotation>

YOLO txt 格式（每行一个框，坐标归一化到 0-1）:
    class_id  x_center  y_center  width  height

用法:
    # 基本转换（xml 与图片同目录，txt 写到 xml 旁边）
    python tools/xml2yolo.py -i ./images

    # xml 单独存放，txt 输出到 labels 目录
    python tools/xml2yolo.py -i ./images -x ./Annotations -l ./labels

    # 指定类别（逗号分隔，顺序即 class_id），并把无标注图片移动到 unlabeled
    python tools/xml2yolo.py -i ./images -c "durian,background" --move
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 可识别的图片格式(大写扩展名)
SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

# ==================== 配置区(直接在这里修改) ====================
IMAGES_DIR = ""          # 图片目录（必填）
XML_DIR = ""             # XML 标注目录；留空表示与图片同目录
LABELS_DIR = ""          # YOLO txt 输出目录；留空表示写到 xml 旁边
UNLABELED_DIR = ""       # 无标注图片输出目录；留空表示 images 同级下的 unlabeled
CLASSES = ""             # 类别：逗号分隔（如 "durian,background"）或 classes.txt 路径；留空自动收集
MOVE_UNLABELED = False   # True: 移动无标注图片；False: 复制
NO_UNLABELED = False     # True: 不筛选无标注图片
# ================================================================


def parse_xml(path: Path):
    """解析单个 XML，返回 (filename, width, height, [(name, xmin, ymin, xmax, ymax), ...])。"""
    root = ET.parse(path).getroot()

    filename = (root.findtext("filename") or "").strip()

    width = height = None
    size_el = root.find("size")
    if size_el is not None:
        try:
            width = int(float(size_el.findtext("width").strip()))
            height = int(float(size_el.findtext("height").strip()))
        except (AttributeError, TypeError, ValueError):
            width = height = None

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))
        except (TypeError, ValueError):
            # 兼容 x1/y1/x2/y2 形式的标注
            try:
                xmin = float(bbox.findtext("x1"))
                ymin = float(bbox.findtext("y1"))
                xmax = float(bbox.findtext("x2"))
                ymax = float(bbox.findtext("y2"))
            except (TypeError, ValueError):
                continue
        if name:
            boxes.append((name, xmin, ymin, xmax, ymax))
    return filename, width, height, boxes


def read_image_size(path: Path):
    """从图片读取 (width, height)，用于 XML 缺失 size 时回退。"""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def to_yolo_line(name, xmin, ymin, xmax, ymax, width, height, class_map):
    """把一个 bndbox 转成 YOLO 行；类别不在映射中返回 None。"""
    cls = class_map.get(name)
    if cls is None:
        return None
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    # 防越界
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def resolve_classes(records, classes_arg):
    """确定类别映射 {name: id}；支持逗号分隔 / classes.txt 文件 / 自动收集。"""
    if classes_arg:
        p = Path(classes_arg)
        if p.is_file():
            names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            names = [x.strip() for x in classes_arg.split(",") if x.strip()]
        return {name: i for i, name in enumerate(names)}

    # 自动收集：按 XML 中首次出现顺序
    names = []
    for _, _, _, _, boxes in records:
        for name, *_ in boxes:
            if name and name not in names:
                names.append(name)
    return {name: i for i, name in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="XML 标注转 YOLO 格式，并筛选无标注图片")
    parser.add_argument("--images", "-i", default=IMAGES_DIR, help="图片目录（必填）")
    parser.add_argument("--xml", "-x", default=XML_DIR, help="XML 标注目录（默认与图片同目录）")
    parser.add_argument("--labels", "-l", default=LABELS_DIR, help="YOLO txt 输出目录（默认写到 xml 旁边）")
    parser.add_argument("--unlabeled", "-u", default=UNLABELED_DIR, help="无标注图片输出目录（默认 images 同级下的 unlabeled）")
    parser.add_argument("--classes", "-c", default=CLASSES, help="类别：逗号分隔或 classes.txt 路径；留空自动收集")
    parser.add_argument("--move", action="store_true", default=MOVE_UNLABELED, help="移动无标注图片（默认复制）")
    parser.add_argument("--no-unlabeled", action="store_true", default=NO_UNLABELED, help="不筛选无标注图片")
    args = parser.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"图片目录不存在: {images_dir}")

    xml_dir = Path(args.xml).expanduser().resolve() if args.xml else images_dir
    if not xml_dir.is_dir():
        raise SystemExit(f"XML 目录不存在: {xml_dir}")

    # 1. 收集图片（stem -> 路径）
    images = {
        p.stem: p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
    }
    if not images:
        raise SystemExit("图片目录中没有找到图片。")

    # 2. 收集 XML
    xml_files = sorted(
        p for p in xml_dir.rglob("*")
        if p.is_file() and p.suffix.upper() == ".XML"
    )
    if not xml_files:
        raise SystemExit("没有找到 XML 标注文件。")

    # 3. 解析所有 XML
    records = []
    for xml_path in xml_files:
        try:
            records.append((xml_path, *parse_xml(xml_path)))
        except ET.ParseError as exc:
            print(f"[跳过] XML 解析失败 {xml_path}: {exc}")

    # 4. 类别映射
    class_map = resolve_classes(records, args.classes)

    labels_dir = Path(args.labels).expanduser() if args.labels else None
    if labels_dir:
        labels_dir.mkdir(parents=True, exist_ok=True)

    # 5. 转换并写 txt
    ok_count = fail_count = 0
    annotated_stems = set()
    for xml_path, filename, width, height, boxes in records:
        annotated_stems.add(xml_path.stem)
        if filename:
            annotated_stems.add(Path(filename).stem)

        # XML 缺少 size 时从图片回退读取尺寸
        if not width or not height:
            img_path = images.get(xml_path.stem)
            if img_path:
                try:
                    width, height = read_image_size(img_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[失败] 无法读取图片尺寸 {img_path}: {exc}")

        if not width or not height:
            fail_count += 1
            print(f"[失败] 缺少尺寸信息: {xml_path}")
            continue

        lines = []
        for name, xmin, ymin, xmax, ymax in boxes:
            line = to_yolo_line(name, xmin, ymin, xmax, ymax, width, height, class_map)
            if line is not None:
                lines.append(line)
            else:
                print(f"[警告] 未知类别 '{name}'（{xml_path.name}），已忽略")

        dst = (labels_dir / (xml_path.stem + ".txt")) if labels_dir else xml_path.with_suffix(".txt")
        dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        ok_count += 1
        print(f"[成功] {xml_path.name} -> {dst.name}（{len(lines)} 个框）")

    # 6. 保存 classes.txt
    classes_out = (labels_dir if labels_dir else xml_dir) / "classes.txt"
    classes_out.write_text(
        "\n".join(name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])) + "\n",
        encoding="utf-8",
    )
    print(f"类别映射（共 {len(class_map)} 类）: {class_map}")
    print(f"已保存: {classes_out}")

    # 7. 筛选无标注图片
    if not args.no_unlabeled:
        unlabeled = sorted(
            (img for stem, img in images.items() if stem not in annotated_stems),
            key=lambda p: str(p),
        )
        if unlabeled:
            unlabeled_dir = (
                Path(args.unlabeled).expanduser().resolve()
                if args.unlabeled else images_dir.parent / "unlabeled"
            )
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for img in unlabeled:
                dst = unlabeled_dir / img.name
                if dst.exists():
                    print(f"[跳过] 目标已存在: {dst}")
                    continue
                if args.move:
                    shutil.move(str(img), str(dst))
                else:
                    shutil.copy2(img, dst)
                moved += 1
            print(f"\n无标注图片 {len(unlabeled)} 张，已{'移动' if args.move else '复制'} {moved} 张到: {unlabeled_dir}")
        else:
            print("\n所有图片都有标注。")

    print(f"\n完成：转换 {ok_count} 张标注，失败 {fail_count} 张。")


if __name__ == "__main__":
    main()
