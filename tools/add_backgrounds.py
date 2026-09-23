"""
背景图添加命令行工具：把背景图作为负样本加入数据集（降低误检）。

与前端模块 `QT/modules/background.py` 对应，逻辑保持一致。
自动：随机抽取背景图、缩放到目标分辨率、复制到 images、生成同名空标签文件。

用法:
    python tools/add_backgrounds.py -b ./backgrounds -d ./dataset -n 60 -s 1280
    python tools/add_backgrounds.py -b ./backgrounds -d ./dataset -n 60 -v 15 -s 1280
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


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


def resize_to_target(img_bgr, target):
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= 0 or longest == target:
        return img_bgr
    scale = target / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def detect_structure(dataset_dir):
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        return None
    if (d / "images" / "train").is_dir():
        return {
            "split": True,
            "train_images": d / "images" / "train",
            "train_labels": d / "labels" / "train",
            "val_images": d / "images" / "val" if (d / "images" / "val").is_dir() else None,
            "val_labels": d / "labels" / "val" if (d / "labels" / "val").is_dir() else None,
        }
    if (d / "images").is_dir():
        return {
            "split": False,
            "train_images": d / "images",
            "train_labels": d / "labels",
            "val_images": None,
            "val_labels": None,
        }
    return None


def next_bg_index(images_dir):
    idx = 1
    for p in images_dir.glob("bg_*"):
        try:
            num = int(p.stem[3:])
            idx = max(idx, num + 1)
        except ValueError:
            continue
    return idx


def add_to(bg_pool, images_dir, labels_dir, target_size, tag):
    if not images_dir:
        return 0
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    idx = next_bg_index(images_dir)
    ok = 0
    for bg in bg_pool:
        img = imread_unicode(bg)
        if img is None:
            print(f"[失败] 无法读取背景图: {bg}")
            continue
        resized = resize_to_target(img, target_size)
        dst = images_dir / f"bg_{idx:03d}{bg.suffix.lower() or '.jpg'}"
        if not imwrite_unicode(dst, resized):
            print(f"[失败] 保存失败: {dst}")
            continue
        (labels_dir / f"bg_{idx:03d}.txt").touch()
        print(f"[{tag}] {bg.name} -> {dst.name}（空标签已生成）")
        ok += 1
        idx += 1
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="把背景图作为负样本加入数据集")
    parser.add_argument("--bg", "-b", required=True, help="背景图文件夹")
    parser.add_argument("--dataset", "-d", required=True, help="数据集根目录")
    parser.add_argument("--train-count", "-n", type=int, default=60, help="训练集背景图数量，默认 60")
    parser.add_argument("--val-count", "-v", type=int, default=0, help="验证集背景图数量，默认 0")
    parser.add_argument("--size", "-s", type=int, default=1280, help="目标分辨率，默认 1280")
    args = parser.parse_args()

    bg_dir = Path(args.bg).expanduser().resolve()
    if not bg_dir.is_dir():
        raise SystemExit(f"背景图文件夹不存在: {bg_dir}")

    structure = detect_structure(args.dataset)
    if structure is None:
        raise SystemExit(f"数据集目录不存在或结构无法识别: {args.dataset}")

    bg_files = sorted(p for p in bg_dir.rglob("*")
                      if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS)
    if not bg_files:
        raise SystemExit("背景图文件夹中没有找到图片。")

    need = args.train_count + args.val_count
    if len(bg_files) < need:
        print(f"背景图只有 {len(bg_files)} 张，少于请求的 {need} 张，将全部使用。")
        need = len(bg_files)

    random.shuffle(bg_files)
    train_pool = bg_files[:args.train_count]
    val_pool = bg_files[args.train_count:args.train_count + args.val_count]

    print(f"数据集结构：{'已划分（train/val）' if structure['split'] else '未划分（平铺）'}；"
          f"目标分辨率 {args.size}；训练集 {len(train_pool)} 张，验证集 {len(val_pool)} 张。")

    ok = add_to(train_pool, structure["train_images"], structure["train_labels"],
                args.size, "train")
    if val_pool and structure["val_images"] and structure["val_labels"]:
        ok += add_to(val_pool, structure["val_images"], structure["val_labels"],
                     args.size, "val")
    elif val_pool:
        print("提示：数据集未划分或没有 val 目录，验证集背景图未添加。")

    print(f"\n完成: 成功添加 {ok} 张背景图。")


if __name__ == "__main__":
    main()
