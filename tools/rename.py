"""
图片重命名命令行工具：把图片按序号从指定起点开始连续命名。

与前端模块 `QT/modules/rename.py` 对应，逻辑保持一致。
命名格式：{前缀}{序号:0{位数}d}{原扩展名}
支持原地重命名（--inplace）或复制到新文件夹（-o）。

用法:
    python tools/rename.py -i ./images -o ./renamed --start 1 --width 4 --prefix img_
    python tools/rename.py -i ./images --inplace --start 100 --width 3
"""

import argparse
import re
import shutil
from pathlib import Path

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def _find_labels_dir(images_dir):
    p = Path(images_dir).expanduser()
    if p.name != "images":
        candidate = p.parent.parent / "labels" / p.name
        if candidate.is_dir():
            return candidate
    candidate = p.parent / "labels"
    if candidate.is_dir():
        return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="把图片按序号连续重命名")
    parser.add_argument("--images", "-i", required=True, help="图片文件夹")
    parser.add_argument("--out", "-o", default="", help="输出目录（复制模式，默认不指定则需 --inplace）")
    parser.add_argument("--inplace", action="store_true", help="原地重命名（直接改原文件）")
    parser.add_argument("--labels", "-l", default="", help="标签目录（留空自动检测 images 同级 labels）")
    parser.add_argument("--start", type=int, default=1, help="起始序号，默认 1")
    parser.add_argument("--width", type=int, default=4, help="补零位数，默认 4")
    parser.add_argument("--prefix", default="", help="前缀，默认空")
    parser.add_argument("--sort", choices=["name", "mtime"], default="name",
                        help="排序方式：name=文件名自然排序，mtime=修改时间")
    parser.add_argument("--no-labels", action="store_true", help="不同步重命名标签")
    args = parser.parse_args()

    images = Path(args.images).expanduser()
    if not images.is_dir():
        raise SystemExit(f"图片文件夹不存在: {images}")

    files = [p for p in images.rglob("*")
             if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS]
    if not files:
        raise SystemExit("图片文件夹中没有找到图片。")

    if args.sort == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: _natural_key(p.name))

    inplace = args.inplace
    out = Path(args.out).expanduser() if args.out else None
    if not inplace and out is None:
        raise SystemExit("请指定 --inplace（原地重命名）或 -o（输出目录）。")
    if not inplace and out.resolve() == images.resolve():
        raise SystemExit("输出目录不能与图片文件夹相同。")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    labels = None
    if not args.no_labels:
        labels = Path(args.labels).expanduser() if args.labels else _find_labels_dir(images)
        if labels is not None and not labels.is_dir():
            print(f"标签目录不存在（将不同步标签）：{labels}")
            labels = None
        if labels is not None:
            print(f"同步标签目录：{labels}")

    print(f"共 {len(files)} 张图片，起始序号 {args.start}，补零 {args.width} 位，"
          f"前缀 {args.prefix or '（无）'}，方式：{'原地重命名' if inplace else '复制到新文件夹'}。")

    ok = 0
    fail = 0
    for i, src in enumerate(files):
        seq = args.start + i
        new_stem = f"{args.prefix}{seq:0{args.width}d}"
        ext = src.suffix.lower() or ".jpg"

        dst_img = src.with_name(new_stem + ext) if inplace else out / (new_stem + ext)
        if dst_img.exists() and dst_img.resolve() != src.resolve():
            print(f"[跳过] 目标已存在：{dst_img.name}（来自 {src.name}）")
            fail += 1
            continue

        try:
            if inplace:
                shutil.move(str(src), str(dst_img))
            else:
                shutil.copy2(src, dst_img)
            if labels is not None:
                src_txt = labels / (src.stem + ".txt")
                if src_txt.is_file():
                    if inplace:
                        shutil.move(str(src_txt), str(labels / (new_stem + ".txt")))
                    else:
                        out_labels = out.parent / "labels" if out.name == "images" else out / "labels"
                        out_labels.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_txt, out_labels / (new_stem + ".txt"))
            print(f"[{i + 1}/{len(files)}] {src.name} -> {dst_img.name}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {src.name}：{exc}")
            fail += 1

    print(f"\n完成: 成功重命名 {ok} 张，失败/跳过 {fail} 张。")


if __name__ == "__main__":
    main()
