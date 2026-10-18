"""
把 ISAT（Image Semantic Annotation Tool）格式的 JSON 标注转换成 YOLO 训练格式的 txt，
并把没有标注的图片筛选出来复制/移动到单独文件夹。

ISAT JSON 格式示例（每个图片对应一个同名 .json）:
    {
        "info": {
            "name": "xxx.jpg",
            "width": 1920,
            "height": 1080
        },
        "objects": [
            {
                "category": "durian",
                "group": 1,
                "segmentation": [[x1, y1, x2, y2, ...]],
                "area": 1234,
                "layer": 1,
                "bbox": [x, y, w, h]
            }
        ]
    }

说明：
- info.name 为图片文件名，width/height 为图片尺寸（像素）；
- objects[].bbox 为外接框 [左上角 x, 左上角 y, 宽, 高]（像素坐标）；
- objects[].category 为类别名，映射到 class_id。

YOLO txt 格式（每行一个框，坐标归一化到 0-1）:
    class_id  x_center  y_center  width  height

用法:
    # 基本转换（json 与图片同目录，txt 写到 json 旁边）
    python tools/isat2yolo.py -i ./images

    # json 单独存放，txt 输出到 labels 目录
    python tools/isat2yolo.py -i ./images -j ./jsons -l ./labels

    # 指定类别（逗号分隔，顺序即 class_id），并把无标注图片移动到 unlabeled
    python tools/isat2yolo.py -i ./images -j ./jsons -c "durian,background" --move
"""

import argparse
import json
import re
import shutil
from pathlib import Path

# 可识别的图片格式（大写扩展名）
SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

# ==================== 配置区（直接在这里修改） ====================
IMAGES_DIR = ""          # 图片目录（必填）
JSON_DIR = ""            # ISAT JSON 标注目录；留空表示与图片同目录
LABELS_DIR = ""          # YOLO txt 输出目录；留空表示写到 json 旁边
UNLABELED_DIR = ""       # 无标注图片输出目录；留空表示 images 同级下的 unlabeled
CLASSES = ""             # 类别：逗号分隔（如 "durian,background"）或 classes.txt 路径；留空自动收集
MOVE_UNLABELED = False   # True: 移动无标注图片；False: 复制
NO_UNLABELED = False     # True: 不筛选无标注图片
# ================================================================


def _flatten_numbers(seg):
    """递归地把 segmentation 里嵌套的数字全部展开为一维浮点数列表。"""
    nums = []
    if isinstance(seg, (int, float)):
        return [float(seg)]
    if isinstance(seg, (list, tuple)):
        for item in seg:
            nums.extend(_flatten_numbers(item))
    return nums


def _seg_bbox(seg):
    """从 segmentation 多边形计算外接框 (xmin, ymin, xmax, ymax)；失败返回 None。"""
    nums = _flatten_numbers(seg)
    if len(nums) < 4 or len(nums) % 2 != 0:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def parse_isat(path: Path):
    """解析单个 ISAT JSON，返回 (filename, width, height, [(name, xmin, ymin, xmax, ymax), ...])。

    ISAT 标注的核心是 objects[].segmentation（多边形），bbox 为辅助字段且是
    [x1, y1, x2, y2]（左上/右下两点式）。这里优先用 segmentation 计算外接框，
    segmentation 缺失时回退到 bbox。
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    info = data.get("info") or {}
    filename = (info.get("name") or "").strip()

    width = info.get("width")
    height = info.get("height")
    try:
        width = int(float(width)) if width is not None else None
        height = int(float(height)) if height is not None else None
    except (TypeError, ValueError):
        width = height = None

    boxes = []
    for obj in data.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        name = (obj.get("category") or "").strip()

        # 优先：segmentation 多边形 -> 外接框
        xmin = ymin = xmax = ymax = None
        seg = obj.get("segmentation")
        if seg:
            rect = _seg_bbox(seg)
            if rect:
                xmin, ymin, xmax, ymax = rect

        # 回退：bbox（ISAT 为 [x1, y1, x2, y2]）
        if xmin is None:
            bbox = obj.get("bbox")
            if bbox and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                    xmin, ymin, xmax, ymax = x1, y1, x2, y2
                except (TypeError, ValueError):
                    pass

        if xmin is None or not name:
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))
    return filename, width, height, boxes


def read_image_size(path: Path):
    """从图片读取 (width, height)，用于 JSON 缺失 info.width/height 时回退。"""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def to_yolo_line(name, xmin, ymin, xmax, ymax, width, height, class_map):
    """把一个 bbox 转成 YOLO 行；类别不在映射中返回 None。"""
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


def _natural_key(name):
    """自然排序键：把数字段转成整数，使 2-xxx 排在 10-xxx 之前。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def resolve_classes(records, classes_arg):
    """确定类别映射 {name: id}；支持逗号分隔 / classes.txt 文件 / 自动收集。"""
    if classes_arg:
        p = Path(classes_arg)
        if p.is_file():
            names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            names = [x.strip() for x in classes_arg.split(",") if x.strip()]
        return {name: i for i, name in enumerate(names)}

    # 自动收集：按类别名自然排序，保证 class_id 稳定、可预期（而非依赖文件首次出现顺序）
    names = []
    for _, _, _, _, boxes in records:
        for name, *_ in boxes:
            if name and name not in names:
                names.append(name)
    names.sort(key=_natural_key)
    return {name: i for i, name in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="ISAT JSON 标注转 YOLO 格式，并筛选无标注图片")
    parser.add_argument("--images", "-i", default=IMAGES_DIR, help="图片目录（必填）")
    parser.add_argument("--json", "-j", default=JSON_DIR, help="ISAT JSON 标注目录（默认与图片同目录）")
    parser.add_argument("--labels", "-l", default=LABELS_DIR, help="YOLO txt 输出目录（默认写到 json 旁边）")
    parser.add_argument("--unlabeled", "-u", default=UNLABELED_DIR, help="无标注图片输出目录（默认 images 同级下的 unlabeled）")
    parser.add_argument("--classes", "-c", default=CLASSES, help="类别：逗号分隔或 classes.txt 路径；留空自动收集")
    parser.add_argument("--move", action="store_true", default=MOVE_UNLABELED, help="移动无标注图片（默认复制）")
    parser.add_argument("--no-unlabeled", action="store_true", default=NO_UNLABELED, help="不筛选无标注图片")
    args = parser.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"图片目录不存在: {images_dir}")

    json_dir = Path(args.json).expanduser().resolve() if args.json else images_dir
    if not json_dir.is_dir():
        raise SystemExit(f"JSON 目录不存在: {json_dir}")

    # 1. 收集图片（stem -> 路径）
    images = {
        p.stem: p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
    }
    if not images:
        raise SystemExit("图片目录中没有找到图片。")

    # 2. 收集 JSON
    json_files = sorted(
        p for p in json_dir.rglob("*")
        if p.is_file() and p.suffix.upper() == ".JSON"
    )
    if not json_files:
        raise SystemExit("没有找到 ISAT JSON 标注文件。")

    # 3. 解析所有 JSON
    records = []
    failed_stems = set()  # 解析失败的标注文件 stem，避免被误判为「无标注」
    for json_path in json_files:
        try:
            records.append((json_path, *parse_isat(json_path)))
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] JSON 解析失败 {json_path}: {exc}")
            failed_stems.add(json_path.stem)

    # 4. 类别映射
    class_map = resolve_classes(records, args.classes)

    labels_dir = Path(args.labels).expanduser() if args.labels else None
    if labels_dir:
        labels_dir.mkdir(parents=True, exist_ok=True)

    # 5. 转换并写 txt
    ok_count = fail_count = 0
    annotated_stems = set(failed_stems)  # 解析失败的文件不参与无标注筛选
    for json_path, filename, width, height, boxes in records:
        # JSON 缺少尺寸时从图片回退读取
        if not width or not height:
            img_path = images.get(json_path.stem)
            if img_path:
                try:
                    width, height = read_image_size(img_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[失败] 无法读取图片尺寸 {img_path}: {exc}")

        if not width or not height:
            print(f"[失败] 缺少尺寸信息：{json_path}")
            fail_count += 1
            continue

        lines = []
        for name, xmin, ymin, xmax, ymax in boxes:
            line = to_yolo_line(name, xmin, ymin, xmax, ymax, width, height, class_map)
            if line is not None:
                lines.append(line)
            else:
                print(f"[警告] 未知类别 '{name}'（{json_path.name}），已忽略")

        if lines:
            # 有有效框 → 已标注
            dst = (labels_dir / (json_path.stem + ".txt")) if labels_dir else json_path.with_suffix(".txt")
            dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
            ok_count += 1
            annotated_stems.add(json_path.stem)
            if filename:
                annotated_stems.add(Path(filename).stem)
            print(f"[成功] {json_path.name} -> {dst.name}（{len(lines)} 个框）")
        else:
            # 0 个有效框 → 视为无标注（与分析口径一致）
            print(f"[无标注] {json_path.name}（0 个有效框）")

    # 保存 classes.txt
    classes_out = (labels_dir if labels_dir else json_dir) / "classes.txt"
    classes_out.write_text(
        "\n".join(name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])) + "\n",
        encoding="utf-8",
    )
    print(f"类别映射（共 {len(class_map)} 类）：{class_map}")
    print(f"已保存：{classes_out}")

    # 筛选无标注图片
    if not args.no_unlabeled:
        unlabeled = sorted(
            (img for stem, img in images.items() if stem not in annotated_stems),
            key=lambda p: str(p),
        )
        if unlabeled:
            unlabeled_dir = (
                Path(args.unlabeled).expanduser()
                if args.unlabeled else images_dir.parent / "unlabeled"
            )
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for img in unlabeled:
                dst = unlabeled_dir / img.name
                if dst.exists():
                    print(f"[跳过] 目标已存在：{dst}")
                    continue
                if args.move:
                    shutil.move(str(img), str(dst))
                else:
                    shutil.copy2(img, dst)
                moved += 1
            action = "移动" if args.move else "复制"
            print(f"无标注图片 {len(unlabeled)} 张，已{action} {moved} 张到：{unlabeled_dir}")
        else:
            print("所有图片都有标注。")

    print(f"\n完成：转换 {ok_count} 张标注，失败 {fail_count} 张。")


if __name__ == "__main__":
    main()
