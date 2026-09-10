"""
数据增强命令行工具：对图片（及对应 YOLO 标签）批量做数据增强。

与前端模块 `QT/modules/augment.py` 对应，算法保持一致。
每个启用的增强方式对每张原图生成一张新图（命名 `<原名>_<tag>.<后缀>`），
提供 labels 目录时同步生成同名 `.txt` 标签。

用法:
    python tools/augment.py -i ./images -o ./aug_out --hflip --vflip --rot90
    python tools/augment.py -i ./images -l ./labels -o ./aug_out \
        --brightness 0.8 --contrast 0.8 --saturation 1.2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

# 几何增强：参数名 -> 显示名（会同步变换标签坐标）
GEO_FLAGS = {
    "hflip": "水平翻转",
    "vflip": "垂直翻转",
    "rot90": "旋转 90°（顺时针）",
    "rot180": "旋转 180°",
    "rot270": "旋转 270°（逆时针）",
}

# 色彩 / 噪声增强：参数名 -> (显示名, 默认值)
COLOR_ARGS = {
    "brightness": "亮度调整",
    "contrast": "对比度调整",
    "saturation": "饱和度调整",
    "hue": "色调偏移",
    "gaussian": "高斯噪声",
    "saltpepper": "椒盐噪声",
}

GEO_KEYS = set(GEO_FLAGS)


def imread_unicode(path):
    """读取图片(支持中文等非 ASCII 路径)。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片(支持中文等非 ASCII 路径)。"""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------- 增强函数（与 QT/modules/augment.py 保持一致） ----------
def augment_hflip(img):
    return cv2.flip(img, 1)


def augment_vflip(img):
    return cv2.flip(img, 0)


def augment_rot90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def augment_rot180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def augment_rot270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def augment_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_contrast(img, factor):
    out = (img.astype(np.float32) - 127.5) * factor + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_saturation(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_hue(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + delta) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_salt_pepper(img, prob):
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < prob / 2] = 255
    out[mask > 1 - prob / 2] = 0
    return out


APPLY = {
    "hflip": lambda img, _p: augment_hflip(img),
    "vflip": lambda img, _p: augment_vflip(img),
    "rot90": lambda img, _p: augment_rot90(img),
    "rot180": lambda img, _p: augment_rot180(img),
    "rot270": lambda img, _p: augment_rot270(img),
    "brightness": augment_brightness,
    "contrast": augment_contrast,
    "saturation": augment_saturation,
    "hue": augment_hue,
    "gaussian": augment_gaussian_noise,
    "saltpepper": augment_salt_pepper,
}


def transform_yolo_line(line, key):
    """对 YOLO 标签一行做几何变换（与前端模块一致）。"""
    parts = line.split()
    if len(parts) < 5:
        return line
    cls = parts[0]
    try:
        cx, cy, w, h = (float(x) for x in parts[1:5])
    except ValueError:
        return line

    if key == "hflip":
        cx = 1.0 - cx
    elif key == "vflip":
        cy = 1.0 - cy
    elif key == "rot90":
        cx, cy, w, h = 1.0 - cy, cx, h, w
    elif key == "rot180":
        cx, cy = 1.0 - cx, 1.0 - cy
    elif key == "rot270":
        cx, cy, w, h = cy, 1.0 - cx, h, w
    else:
        return line

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="批量数据增强（可同步 YOLO 标签）")
    parser.add_argument("--input", "-i", required=True, help="图片目录或单个图片文件")
    parser.add_argument("--labels", "-l", default="", help="labels 目录（可选，与图片同名的 .txt）")
    parser.add_argument("--output", "-o", required=True, help="输出目录")

    # 几何增强（布尔开关）
    for key in GEO_FLAGS:
        parser.add_argument(f"--{key}", action="store_true", help=GEO_FLAGS[key])

    # 色彩 / 噪声增强（传值即启用）
    parser.add_argument("--brightness", type=float, default=None, help="亮度因子，如 0.8")
    parser.add_argument("--contrast", type=float, default=None, help="对比度因子，如 0.8")
    parser.add_argument("--saturation", type=float, default=None, help="饱和度因子，如 1.2")
    parser.add_argument("--hue", type=float, default=None, help="色调偏移，如 10")
    parser.add_argument("--gaussian", type=float, default=None, help="高斯噪声强度，如 15")
    parser.add_argument("--saltpepper", type=float, default=None, help="椒盐噪声比例，如 0.01")

    args = parser.parse_args()

    # 汇总启用的增强方式
    ops = []
    for key, label in GEO_FLAGS.items():
        if getattr(args, key):
            ops.append((key, label, None))
    for key, label in COLOR_ARGS.items():
        val = getattr(args, key)
        if val is not None:
            ops.append((key, label, val))

    if not ops:
        raise SystemExit("请至少启用一种增强方式（如 --hflip 或 --brightness 0.8）。")

    src_path = Path(args.input).expanduser().resolve()
    if not src_path.exists():
        raise SystemExit(f"输入路径不存在: {src_path}")

    input_is_file = src_path.is_file()
    if input_is_file:
        if src_path.suffix.upper() not in SUPPORTED_EXTS:
            raise SystemExit(f"输入文件格式不支持: {src_path}")
        files = [src_path]
    else:
        files = sorted(
            p for p in src_path.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )

    if not files:
        raise SystemExit("没有找到可处理的图片。")

    labels_dir = Path(args.labels).expanduser().resolve() if args.labels else None
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"共 {len(files)} 张图片，启用 {len(ops)} 种增强："
          + "、".join(label for _, label, _ in ops))

    ok = fail = 0
    for src in files:
        img = imread_unicode(src)
        if img is None:
            print(f"[失败] 无法读取: {src}")
            fail += 1
            continue

        rel = src.name if input_is_file else src.relative_to(src_path)

        if labels_dir is not None:
            label_src = labels_dir / rel.with_suffix(".txt")
        else:
            label_src = src.with_suffix(".txt")
        label_lines = None
        if label_src.is_file():
            label_lines = label_src.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()

        for key, label, param in ops:
            result = APPLY[key](img, param)

            dst = out / rel.parent / (rel.stem + "_" + key + rel.suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)

            if label_lines is not None:
                if key in GEO_KEYS:
                    new_lines = [transform_yolo_line(line, key) for line in label_lines]
                else:
                    new_lines = label_lines[:]
                dst.with_suffix(".txt").write_text(
                    "\n".join(new_lines) + ("\n" if new_lines else ""),
                    encoding="utf-8",
                )

            if imwrite_unicode(dst, result):
                ok += 1
                print(f"[成功] {src.name} --{label}--> {dst.name}")
            else:
                fail += 1
                print(f"[失败] 保存失败: {src}（{label}）")

    print(f"\n完成: 成功 {ok} 张, 失败 {fail} 张。输出目录: {out}")


if __name__ == "__main__":
    main()
