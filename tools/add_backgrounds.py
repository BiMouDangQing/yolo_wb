"""
背景图添加命令行工具：把背景图作为负样本加入数据集（降低误检）。

与前端模块 `QT/modules/background.py` 对应，逻辑保持一致。
根据「背景占比」自动计算需要添加的背景图数量，背景图不足时用随机数据增强扩充，
缩放到目标分辨率、复制到 images、生成同名空标签文件。

用法:
    python tools/add_backgrounds.py -b ./backgrounds -d ./dataset -r 10 -s 1280
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


def random_augment(img_bgr):
    """对背景图随机应用数据增强（翻转/旋转/亮度/对比度/噪声），返回新图。"""
    img = img_bgr.copy()
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.2:
        img = cv2.flip(img, 0)
    rot = random.choice([
        None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ])
    if rot is not None:
        img = cv2.rotate(img, rot)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = np.clip(
            (img.astype(np.float32) - 127.5) * factor + 127.5, 0, 255
        ).astype(np.uint8)
    if random.random() < 0.3:
        sigma = random.uniform(3, 12)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def detect_structure(dataset_dir):
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        return None
    if (d / "images" / "train").is_dir():
        return {
            "split": True,
            "train_images": d / "images" / "train",
            "train_labels": d / "labels" / "train",
        }
    if (d / "images").is_dir():
        return {
            "split": False,
            "train_images": d / "images",
            "train_labels": d / "labels",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="把背景图作为负样本加入数据集")
    parser.add_argument("--bg", "-b", required=True, help="背景图文件夹")
    parser.add_argument("--dataset", "-d", required=True, help="数据集根目录")
    parser.add_argument("--ratio", "-r", type=int, default=10,
                        help="背景占比（百分比，默认 10，建议 5~10）")
    parser.add_argument("--size", "-s", type=int, default=1280,
                        help="目标分辨率，默认 1280")
    args = parser.parse_args()

    bg_dir = Path(args.bg).expanduser().resolve()
    if not bg_dir.is_dir():
        raise SystemExit(f"背景图文件夹不存在: {bg_dir}")

    structure = detect_structure(args.dataset)
    if structure is None:
        raise SystemExit(f"数据集目录不存在或结构无法识别: {args.dataset}")

    images_dir = structure["train_images"]
    labels_dir = structure["train_labels"]
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        and not p.stem.startswith("bg_")
    )
    n = len(existing)
    if n == 0:
        raise SystemExit("数据集中没有找到图片。")

    ratio = max(0, min(args.ratio, 49))
    need = int(round(n * ratio / (100 - ratio)))
    print(f"原图 {n} 张，背景占比 {ratio}%，需要背景图 {need} 张。")

    bg_files = sorted(p for p in bg_dir.rglob("*")
                      if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS)
    if not bg_files:
        raise SystemExit("背景图文件夹中没有找到图片。")

    random.shuffle(bg_files)
    if len(bg_files) < need:
        print(f"背景图只有 {len(bg_files)} 张，将通过随机数据增强扩充到 {need} 张。")

    idx = next_bg_index(images_dir)
    ok = 0
    fail = 0
    for i in range(need):
        src = bg_files[i % len(bg_files)]
        img = imread_unicode(src)
        if img is None:
            print(f"[失败] 无法读取背景图: {src}")
            fail += 1
            continue
        if i >= len(bg_files):
            img = random_augment(img)
        resized = resize_to_target(img, args.size)
        dst = images_dir / f"bg_{idx:03d}{src.suffix.lower() or '.jpg'}"
        if not imwrite_unicode(dst, resized):
            print(f"[失败] 保存失败: {dst}")
            fail += 1
            continue
        (labels_dir / f"bg_{idx:03d}.txt").touch()
        print(f"[背景] {src.name}{'（增强）' if i >= len(bg_files) else ''} "
              f"-> {dst.name}（空标签已生成）")
        ok += 1
        idx += 1

    print(f"\n完成: 成功添加 {ok} 张背景图，失败 {fail} 张。")


if __name__ == "__main__":
    main()
