"""
数据集汇总命令行工具：把训练集 / 验证集 / 测试集三个路径下的图片与标签，整理到标准 YOLO 结构。

与前端模块 `QT/modules/assemble.py` 对应，逻辑保持一致。

用法:
    python tools/assemble.py --train ./train --val ./val --test ./test -o ./dataset --nc 1 --names class0
"""

import argparse
import json
import shutil
from pathlib import Path

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def _transfer(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _find_label(p, root):
    txt = p.with_suffix(".txt")
    if txt.is_file():
        return txt
    rel_parts = list(p.relative_to(root).parts)
    if "images" in rel_parts:
        idx = rel_parts.index("images")
        rel_parts[idx] = "labels"
        candidate = root.joinpath(*rel_parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    candidate = root / "labels" / p.relative_to(root).with_suffix(".txt")
    if candidate.is_file():
        return candidate
    return None


def _unique_name(name, used):
    if name not in used:
        return name
    stem, dot, ext = name.rpartition(".")
    stem = stem or name
    ext = ("." + ext) if dot else ""
    i = 2
    while f"{stem}_{i}{ext}" in used:
        i += 1
    return f"{stem}_{i}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="把 train/val/test 三路径图片与标签汇总为标准 YOLO 结构")
    parser.add_argument("--train", default="", help="训练集文件夹")
    parser.add_argument("--val", default="", help="验证集文件夹")
    parser.add_argument("--test", default="", help="测试集文件夹（可选）")
    parser.add_argument("--out", "-o", required=True, help="输出目录")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="copy=复制，move=移动")
    parser.add_argument("--no-yaml", action="store_true", help="不生成 data.yaml")
    parser.add_argument("--nc", type=int, default=1, help="类别数，默认 1")
    parser.add_argument("--names", default="class0", help="类别名（逗号分隔）")
    args = parser.parse_args()

    if not (args.train or args.val or args.test):
        raise SystemExit("训练集、验证集、测试集至少填一个。")

    subsets = [
        ("train", args.train),
        ("val", args.val),
        ("test", args.test),
    ]
    out = Path(args.out).expanduser()

    total = 0
    collected = []
    for subset, d in subsets:
        root = Path(d).expanduser() if d else None
        files = []
        if root is not None and root.is_dir():
            files = sorted(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
            )
        elif d:
            print(f"[{subset}] 路径不存在，跳过：{root}")
        collected.append((subset, root, files))
        total += len(files)

    if total == 0:
        raise SystemExit("三个路径下都没有找到图片。")

    print(f"开始汇总：train/val/test 共 {total} 张图片，"
          f"模式：{'移动' if args.mode == 'move' else '复制'}，输出：{out}")

    done = 0
    missing = 0
    has_labels = False
    for subset, root, files in collected:
        if not files:
            continue
        used_names = set()
        for src in files:
            done += 1
            name = _unique_name(src.name, used_names)
            used_names.add(name)

            img_dst = out / "images" / subset / name
            _transfer(src, img_dst, args.mode)

            lbl = _find_label(src, root)
            if lbl is not None:
                has_labels = True
                lbl_dst = out / "labels" / subset / (name.rsplit(".", 1)[0] + ".txt")
                _transfer(lbl, lbl_dst, args.mode)
            else:
                missing += 1
                print(f"[{subset}] 无标签：{src.name}")

    if not args.no_yaml:
        names = [x.strip() for x in args.names.split(",") if x.strip()]
        if not names:
            names = [f"class{i}" for i in range(args.nc)]
        nc = len(names) if names else args.nc
        lines = [
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
        ]
        if has_labels:
            lines.append("test: images/test")
        lines.append(f"nc: {nc}")
        lines.append(f"names: {json.dumps(names, ensure_ascii=False)}")
        (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已生成 {out / 'data.yaml'}")

    print(f"\n完成: 共汇总 {done} 张图片" +
          (f"，缺失标签 {missing} 个。" if missing else "。"))


if __name__ == "__main__":
    main()
