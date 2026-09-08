"""
将其他格式的图片转换为 YOLO 模型可识别的 JPG 格式。

YOLO(ultralytics) 支持 JPG/PNG/BMP/WebP 等格式,但训练或推理时
统一使用 JPG 最稳妥。本脚本会:

1. 递归扫描输入目录中的所有图片(支持 png/bmp/webp/tif/gif/ppm/ico 等);
2. 统一转换为 RGB 的 JPG 格式(透明背景自动填充为白色);
3. 按原目录结构输出到指定目录(也可原地覆盖)。

用法:
    python tools/jpg.py --input ./images --output ./images_jpg
    python tools/jpg.py --input ./images --output ./images_jpg --quality 95
"""

import argparse
from pathlib import Path

from PIL import Image

# 可识别的输入格式(大写扩展名)
SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
    ".GIF", ".PPM", ".PGM", ".PBM", ".ICO", ".JPE", ".JFIF",
}


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """把各种模式的图片安全地转成 RGB。

    - RGBA / LA / P(带透明):先铺一层白底再合成,避免透明变黑;
    - 灰度 / CMYK 等:直接 convert("RGB")。
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def convert_image(src: Path, dst: Path, quality: int) -> bool:
    """转换单张图片,成功返回 True,失败返回 False。"""
    try:
        with Image.open(src) as img:
            img = convert_to_rgb(img)
            img.save(dst, format="JPEG", quality=quality)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {src} -> {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="把其他格式图片批量转换为 YOLO 可识别的 JPG")
    parser.add_argument("--input", "-i", required=True, help="输入目录(可递归扫描子目录)")
    parser.add_argument("--output", "-o", required=True, help="输出目录(保持原目录结构)")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPG 质量,默认 95")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0

    for src in sorted(input_dir.rglob("*")):
        if not src.is_file() or src.suffix.upper() not in SUPPORTED_EXTS:
            continue

        # 保持相对目录结构,输出统一为 .jpg
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if convert_image(src, dst, args.quality):
            ok_count += 1
            print(f"[成功] {src} -> {dst}")
        else:
            fail_count += 1

    print(f"\n完成:成功 {ok_count} 张,失败 {fail_count} 张。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
