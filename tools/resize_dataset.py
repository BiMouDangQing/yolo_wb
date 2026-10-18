"""
图片缩放命令行工具：把图片缩放到指定尺寸（最长边），生成新数据集，保留 polygon 标签。

与前端模块 `QT/modules/resize.py` 对应，逻辑保持一致。

YOLO（检测 / 分割）标签都是归一化坐标（0~1），图片等比缩放后标签值不变，
因此标签直接原样复制，无需重新计算。

输出结构（保持相对路径镜像）：
    <输出目录>/images/<相对路径>      缩放后的图片
    <输出目录>/labels/<相对路径>      复制的标签（可选）

用法:
    # 缩放图片，自动检测 images 同级 labels 目录
    python tools/resize_dataset.py -i ./dataset/images -o ./dataset_1280 -s 1280

    # 指定 labels 目录
    python tools/resize_dataset.py -i ./dataset/images -l ./dataset/labels -o ./dataset_1280 -s 1280
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片（支持中文等非 ASCII 路径）。"""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def resize_longest(img_bgr, target):
    """把最长边缩放到 target，保持纵横比；已更小或相等则不放大。"""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= 0:
        return img_bgr
    scale = target / longest
    if scale >= 1.0:
        return img_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def find_labels_dir(images_dir):
    """自动检测与 images 对应的 labels 目录，找不到返回 None。"""
    p = Path(images_dir).expanduser()
    candidate = p.parent / "labels"
    if candidate.is_dir():
        return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="把图片缩放到指定尺寸，生成新数据集并保留 polygon 标签")
    parser.add_argument("--images", "-i", required=True, help="图片目录（必填）")
    parser.add_argument("--labels", "-l", default="", help="标签目录（留空自动检测 images 同级 labels）")
    parser.add_argument("--output", "-o", required=True, help="输出目录（新数据集）")
    parser.add_argument("--size", "-s", type=int, default=1280, help="目标尺寸（最长边，默认 1280）")
    parser.add_argument("--no-labels", action="store_true", help="不同步标签")
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

    out = Path(args.output).expanduser().resolve()
    if out == images:
        raise SystemExit("输出目录不能与图片目录相同。")
    out_images = out / "images"
    out_labels = out / "labels"

    labels = None
    if not args.no_labels:
        labels = Path(args.labels).expanduser().resolve() if args.labels else find_labels_dir(images)
        if labels is not None and not labels.is_dir():
            print(f"标签目录不存在（将不同步标签）：{labels}")
            labels = None
        if labels is not None:
            print(f"同步标签目录：{labels}")

    print(f"共 {len(files)} 张图片，目标尺寸（最长边）{args.size}。")

    ok = 0
    fail = 0
    no_label = 0
    for src in files:
        rel = src.relative_to(images)
        img = imread_unicode(src)
        if img is None:
            print(f"[失败] 无法读取 {src.name}")
            fail += 1
            continue
        resized = resize_longest(img, args.size)
        dst = out_images / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(dst, resized):
            print(f"[失败] 保存失败 {dst}")
            fail += 1
            continue

        # 同步标签（归一化坐标不变，直接复制）
        if labels is not None:
            lbl_src = labels / rel.with_suffix(".txt")
            if lbl_src.is_file():
                lbl_dst = out_labels / rel.with_suffix(".txt")
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(lbl_src, lbl_dst)
            else:
                no_label += 1
                print(f"[无标签] {rel}")

        ok += 1
        print(f"[{ok}/{len(files)}] {rel} -> {rel}（{img.shape[1]}x{img.shape[0]} → {resized.shape[1]}x{resized.shape[0]}）")

    print(f"\n完成：缩放 {ok} 张，失败 {fail} 张" +
          (f"，缺标签 {no_label} 张。" if labels is not None else "。"))
    print(f"输出：{out}")


if __name__ == "__main__":
    main()
