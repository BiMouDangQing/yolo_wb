"""
过暗图片校正：提亮暗部、恢复暗部细节。

原理：把图像转到 HSV 色彩空间，对明度通道 V 做 gamma 提亮
（gamma < 1 时整体变亮）。可选用 CLAHE 增强暗部局部对比度。

用法:
    python tools/underexposure.py -i ./images -o ./images_fixed
    python tools/underexposure.py -i ./a.jpg -o ./out --strength 60
    python tools/underexposure.py -i ./a.jpg -o ./out --no-clahe
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
STRENGTH = 50        # 校正强度 0-100，越大提亮越明显
USE_CLAHE = True     # 是否启用 CLAHE 增强暗部细节
# ================================================================


def imread_unicode(path):
    """读取图片(支持中文等非 ASCII 路径)。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片(支持中文等非 ASCII 路径)。

    先写入同目录临时文件，再原子替换到目标路径，
    避免覆盖过程中出错导致原文件损坏或未替换。
    """
    path = Path(path)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    tmp = path.with_name(path.name + ".tmp")
    try:
        buf.tofile(str(tmp))
        tmp.replace(path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def correct_underexposure(img_bgr, strength=50, use_clahe=True):
    """对过暗图片提亮。strength 0~100，越大提亮越明显。"""
    strength = max(0, min(100, strength))
    gamma = max(0.2, 1.0 - strength / 150.0)  # 1.0 ~ 0.33
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    v = np.power(np.clip(v, 0.0, 1.0), gamma)
    hsv[..., 2] = v * 255.0
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if use_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="过暗图片校正")
    parser.add_argument("--input", "-i", default=INPUT_PATH, help="输入目录或单个图片文件")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--strength", "-s", type=int, default=STRENGTH, help="校正强度 0-100，默认 50")
    parser.add_argument("--no-clahe", action="store_true", help="关闭 CLAHE 暗部细节增强")
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

    use_clahe = not args.no_clahe
    ok = fail = 0
    for src in files:
        img = imread_unicode(src)
        if img is None:
            print(f"[失败] 无法读取: {src}")
            fail += 1
            continue
        out = correct_underexposure(img, args.strength, use_clahe)
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
