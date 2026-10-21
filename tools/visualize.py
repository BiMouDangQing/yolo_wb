"""
标签可视化命令行工具：把 YOLO 标签（检测框 / 分割多边形）绘制到图片上。

与前端模块 `QT/modules/visualize.py` 对应，逻辑保持一致。

自动识别标签格式（按每行字段数）：
- 5 字段：检测框  `class cx cy w h`
- >5 奇数：分割多边形 `class x1 y1 x2 y2 ...`

类别名：优先读 labels 同级（或上级）的 classes.txt，找不到则显示 class_id。

用法:
    python tools/visualize.py -i ./dataset/images -l ./dataset/labels -o ./vis
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

# 调色板：给不同类别分配颜色（RGB）
PALETTE = [
    (255, 60, 60), (60, 200, 90), (60, 120, 255), (255, 180, 40),
    (180, 60, 255), (60, 220, 220), (255, 120, 200), (160, 200, 40),
    (240, 140, 60), (100, 160, 255), (200, 80, 80), (80, 200, 140),
]


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def find_font():
    """查找一个支持中文的字体，返回路径；找不到返回 None（用默认字体）。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/simsun.ttc",   # 宋体
        "C:/Windows/Fonts/simkai.ttf",   # 楷体
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def load_font(size):
    path = find_font()
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def find_classes_txt(labels_dir):
    """从 labels 目录向上查找 classes.txt，返回路径或 None。"""
    d = Path(labels_dir).expanduser()
    for _ in range(3):
        p = d / "classes.txt"
        if p.is_file():
            return p
        if d.parent == d:
            break
        d = d.parent
    return None


def load_class_names(labels_dir):
    classes_txt = find_classes_txt(labels_dir)
    if classes_txt is None:
        return []
    try:
        return [
            line.strip()
            for line in classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    except OSError:
        return []


def parse_label_file(path):
    """解析 YOLO 标签，返回 [(cls, kind, values)]。

    kind 为 'detect'（cx cy w h）或 'seg'（顶点坐标列表）。
    """
    items = []
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
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if len(vals) == 4:
                items.append((cls, "detect", vals))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                items.append((cls, "seg", vals))
    except OSError:
        pass
    return items


def draw_labels(img_bgr, items, class_names):
    """把标签绘制到 BGR 图上，返回新 BGR 图。"""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    # 线宽与字号按图片大小自适应
    base = max(h, w)
    line_w = max(2, int(round(base / 640)))
    font_size = max(14, int(round(base / 48)))
    font = load_font(font_size)

    for cls, kind, vals in items:
        color = PALETTE[cls % len(PALETTE)]
        if kind == "detect":
            cx, cy, bw, bh = vals
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            label_pos = (x1, y1 - font_size - 4)
        else:
            pts = []
            for i in range(0, len(vals), 2):
                pts.append((vals[i] * w, vals[i + 1] * h))
            if len(pts) >= 3:
                draw.polygon(pts, outline=color, width=line_w)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label_pos = (min(xs), min(ys) - font_size - 4)

        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        text = f"{cls} {name}"
        # 文字背景框
        bbox = draw.textbbox(label_pos, text, font=font)
        draw.rectangle(bbox, fill=(*color, 220))
        draw.text(label_pos, text, fill=(255, 255, 255), font=font)

    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return out


def find_labels_dir(images_dir):
    p = Path(images_dir).expanduser()
    candidate = p.parent / "labels"
    return candidate if candidate.is_dir() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="把 YOLO 标签绘制到图片上")
    parser.add_argument("--images", "-i", required=True, help="图片目录（必填）")
    parser.add_argument("--labels", "-l", default="", help="标签目录（留空自动检测 images 同级 labels）")
    parser.add_argument("--output", "-o", required=True, help="可视化图片输出目录")
    args = parser.parse_args()

    images = Path(args.images).expanduser().resolve()
    if not images.is_dir():
        raise SystemExit(f"图片目录不存在: {images}")

    files = sorted(
        p for p in images.rglob("*")
        if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
    )
    if not files:
        raise SystemExit("图片目录中没有找到图片。")

    labels = Path(args.labels).expanduser().resolve() if args.labels else find_labels_dir(images)
    if labels is None or not labels.is_dir():
        raise SystemExit(f"标签目录不存在: {labels}")

    class_names = load_class_names(labels)
    if class_names:
        print(f"类别名：{class_names}")

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    no_label = 0
    for src in files:
        rel = src.relative_to(images)
        img = imread_unicode(src)
        if img is None:
            print(f"[失败] 无法读取 {rel}")
            continue

        lbl = labels / rel.with_suffix(".txt")
        items = parse_label_file(lbl) if lbl.is_file() else []
        if not items:
            no_label += 1
            print(f"[无标签] {rel}")

        annotated = draw_labels(img, items, class_names)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(dst, annotated):
            print(f"[失败] 保存失败 {dst}")
            continue
        ok += 1
        print(f"[{ok}/{len(files)}] {rel}（{len(items)} 个标签）")

    print(f"\n完成：可视化 {ok} 张" + (f"，无标签 {no_label} 张。" if no_label else "。"))
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
