"""
把 HEIC / HEIF 图片批量转换为 JPG 格式。

iPhone 拍摄的照片默认就是 HEIC 格式,本脚本依赖 pillow-heif 来解码,
并自动处理 EXIF 旋转方向,避免导出后照片方向错乱。

安装依赖:
    pip install pillow-heif
    # 或
    conda install -c conda-forge pillow-heif

用法:
    1. 直接修改下方【配置区】的 INPUT_PATH / OUTPUT_DIR / OVERWRITE / QUALITY,
       然后运行:  python tools/heic2jpg.py
    2. 也可用命令行参数覆盖配置:
        python tools/heic2jpg.py -i ./photos -o ./photos_jpg
        python tools/heic2jpg.py -i ./photos/IMG_0001.HEIC -o ./out --overwrite
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

try:
    import pillow_heif
except ImportError:
    raise SystemExit("缺少依赖 pillow-heif,请先运行: pip install pillow-heif")

# 注册 HEIF/HEIC 解码器,之后 Image.open 就能直接打开 .heic/.heif 文件
pillow_heif.register_heif_opener()

# HEIC 容器常见的扩展名(可能为大写)
HEIC_EXTS = {".HEIC", ".HEIF", ".HIF", ".AVIF"}

# ==================== 配置区(直接在这里修改) ====================
INPUT_PATH = r"D:\model\data\durian\榴莲"       # 输入目录或单个 HEIC/HEIF 文件
OUTPUT_DIR = "./photos_jpg"   # 输出目录(OVERWRITE 为 True 时忽略)
OVERWRITE = True             # True: 覆盖原图(原地转 JPG 并删除原 HEIC);False: 输出到 OUTPUT_DIR
QUALITY = 95                  # JPG 质量,1-100
# ================================================================


def convert_to_jpg(src: Path, dst: Path, quality: int) -> bool:
    """把单张 HEIC 图片转为 JPG,成功返回 True,失败返回 False。"""
    try:
        with Image.open(src) as img:
            # 按 EXIF 方向旋转(否则 iPhone 照片可能横躺或倒置)
            img = ImageOps.exif_transpose(img)
            # 统一转成 8-bit RGB(HEIC 常为 10-bit,转 8-bit 会轻微损失色深)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dst, format="JPEG", quality=quality)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {src} -> {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="把 HEIC/HEIF 图片批量转换为 JPG")
    parser.add_argument("--input", "-i", default=INPUT_PATH, help="输入目录或单个 HEIC 文件(默认取配置区 INPUT_PATH)")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="输出目录(覆盖原图时忽略,默认取配置区 OUTPUT_DIR)")
    parser.add_argument("--quality", "-q", type=int, default=QUALITY, help="JPG 质量,默认 95")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE, help="覆盖原图:原地转 JPG 并删除原 HEIC 文件")
    args = parser.parse_args()

    src_path = Path(args.input).expanduser().resolve()

    # 收集需要转换的文件
    if src_path.is_file():
        if src_path.suffix.upper() not in HEIC_EXTS:
            raise SystemExit(f"输入文件不是 HEIC/HEIF 文件: {src_path}")
        files = [src_path]
        base_dir = src_path.parent
    elif src_path.is_dir():
        files = sorted(
            p for p in src_path.rglob("*")
            if p.is_file() and p.suffix.upper() in HEIC_EXTS
        )
        base_dir = src_path
    else:
        raise SystemExit(f"输入路径不存在: {src_path}")

    if not files:
        raise SystemExit("没有找到 HEIC/HEIF 文件。")

    # 输出目录:覆盖原图时原地输出,否则写到指定目录
    if args.overwrite:
        output_dir = base_dir
    else:
        output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0

    for src in files:
        if args.overwrite:
            dst = src.with_suffix(".jpg")
        else:
            dst = output_dir / src.with_suffix(".jpg").name

        if convert_to_jpg(src, dst, args.quality):
            ok_count += 1
            print(f"[成功] {src} -> {dst}")
            if args.overwrite:
                try:
                    src.unlink()
                    print(f"       已删除原文件: {src}")
                except OSError as exc:
                    print(f"       删除原文件失败: {exc}")
        else:
            fail_count += 1

    print(f"\n完成:成功 {ok_count} 张,失败 {fail_count} 张。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
