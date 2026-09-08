"""
过曝图片校正：压缩高光、降低整体亮度。

原理：把图像转到 HSV 色彩空间，对明度通道 V 做 gamma 压缩
（gamma > 1 时整体压暗，高光被压缩）。强度越大，压暗越明显。

用法:
    python tools/overexposure.py -i ./images -o ./images_fixed
    python tools/overexposure.py -i ./a.jpg -o ./out --strength 60
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

# ==================== 配置区(直接在这里修改) ====================
INPUT_PATH = ""      # 输入目录或单个图片文件
OUTPUT_DIR = ""      # 输出目录
STRENGTH = 50        # 校正强度 0-100，越大压暗越明显
# ================================================================


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


def correct_overexposure(img_bgr, strength=50):
    """对过曝图片做高光压缩。strength 0~100，越大压暗越明显。"""
    strength = max(0, min(100, strength))
    gamma = 1.0 + strength / 50.0  # 1.0 ~ 3.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    v = np.power(np.clip(v, 0.0, 1.0), gamma)
    hsv[..., 2] = v * 255.0
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="过曝图片校正")
    parser.add_argument("--input", "-i", default=INPUT_PATH, help="输入目录或单个图片文件")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--strength", "-s", type=int, default=STRENGTH, help="校正强度 0-100，默认 50")
    args = parser.parse_args()

    src_path = Path(args.input).expanduser().resolve()
    if not src_path.exists():
        raise SystemExit(f"输入路径不存在: {src_path}")

    if src_path.is_file():
        if src_path.suffix.upper() not in SUPPORTED_EXTS:
            raise SystemExit(f"输入文件格式不支持: {src_path}")
        files = [src_path]
        base_dir = src_path.parent
    else:
        files = sorted(
            p for p in src_path.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        base_dir = src_path

    if not files:
        raise SystemExit("没有找到可处理的图片。")

    if not args.output:
        raise SystemExit("请指定输出目录 -o/--output")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for src in files:
        img = imread_unicode(src)
        if img is None:
            print(f"[失败] 无法读取: {src}")
            fail += 1
            continue
        out = correct_overexposure(img, args.strength)
        rel = src.name if src_path.is_file() else src.relative_to(base_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if imwrite_unicode(dst, out):
            ok += 1
            print(f"[成功] {src} -> {dst}")
        else:
            fail += 1
            print(f"[失败] 保存失败: {src}")

    print(f"\n完成: 成功 {ok} 张, 失败 {fail} 张。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
