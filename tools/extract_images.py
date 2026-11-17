"""
图片提取命令行工具：把一个多层嵌套文件夹里的所有图片提取到目标文件夹。

与前端模块 `QT/modules/extract.py` 对应，逻辑保持一致。

支持：
    - 递归扫描任意层级的子文件夹
    - 扁平输出（默认）或保留子文件夹结构（--keep-structure）
    - 命名冲突处理：rename（默认，自动加序号）/ skip（跳过）/ prefix（加父文件夹名前缀）
    - 可选同时提取同名标签（.txt / .json / .xml / .yaml / .csv）
    - 复制（默认）或移动（--move）

用法:
    # 扁平提取所有图片到 out
    python tools/extract_images.py -i ./A -o ./out

    # 保留子文件夹结构并同时提取标签
    python tools/extract_images.py -i ./A -o ./out --keep-structure --with-labels

    # 同名冲突时加父文件夹名前缀，并移动文件
    python tools/extract_images.py -i ./A -o ./out --conflict prefix --move
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF", ".GIF", ".HEIC",
}

# 可同时提取的标签文件格式
LABEL_EXTS = (".txt", ".json", ".xml", ".yaml", ".yml", ".csv")


def find_images(src, recursive=True):
    """扫描源目录下的所有图片，返回排序后的列表。"""
    src = Path(src).expanduser()
    it = src.rglob("*") if recursive else src.glob("*")
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.upper() in IMAGE_EXTS
    )


def find_label_for(img, labels_dir=None):
    """查找图片对应的标签文件，找不到返回 None。

    查找顺序：
        1. 与图片同目录、同名的标签文件（images 与 labels 混放）
        2. labels_dir 下与图片相对路径同名（且相对路径已去掉 images 层级）的标签
    """
    for ext in LABEL_EXTS:
        sibling = img.with_suffix(ext)
        if sibling.is_file():
            return sibling

    if labels_dir is not None:
        labels_dir = Path(labels_dir).expanduser()
        for ext in LABEL_EXTS:
            # 直接按文件名（去掉 images 层级，如 labels/train/xxx.txt）
            for parent in list(img.parents):
                try:
                    rel_parent = img.relative_to(parent)
                except ValueError:
                    continue
                cand = labels_dir / rel_parent.with_suffix(ext)
                if cand.is_file():
                    return cand
            # 再退化为 labels/xxx.txt
            cand = labels_dir / (img.stem + ext)
            if cand.is_file():
                return cand
    return None


def unique_name(dst_dir, name, used):
    """在 dst_dir 下为 name 生成不冲突的文件名。"""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = name
    i = 1
    while candidate.lower() in used or (dst_dir / candidate).exists():
        candidate = f"{stem}_{i}{suffix}"
        i += 1
    used.add(candidate.lower())
    return candidate


def resolve_conflict(dst_dir, name, strategy, used, img):
    """按冲突策略生成目标文件名。"""
    if strategy == "skip":
        if name.lower() in used or (dst_dir / name).exists():
            return None
        used.add(name.lower())
        return name
    if strategy == "prefix":
        parent = img.parent.name
        stem = Path(name).stem
        suffix = Path(name).suffix
        candidate = f"{parent}_{stem}{suffix}"
        return unique_name(dst_dir, candidate, used)
    # rename（默认）
    return unique_name(dst_dir, name, used)


def main():
    parser = argparse.ArgumentParser(
        description="把多层嵌套文件夹里的图片提取到目标文件夹"
    )
    parser.add_argument("--src", "-i", required=True, help="源文件夹（可含多层子文件夹）")
    parser.add_argument("--out", "-o", required=True, help="目标文件夹")
    parser.add_argument("--no-recursive", action="store_true", help="只扫描顶层，不递归子文件夹")
    parser.add_argument("--keep-structure", action="store_true", help="保留子文件夹结构（默认扁平输出）")
    parser.add_argument("--conflict", choices=["rename", "skip", "prefix"], default="rename",
                        help="同名冲突：rename=自动加序号（默认），skip=跳过，prefix=加父文件夹名前缀")
    parser.add_argument("--with-labels", action="store_true", help="同时提取同名标签文件")
    parser.add_argument("--labels", "-l", default="", help="标签目录（留空则只在图片同目录查找）")
    parser.add_argument("--move", action="store_true", help="移动文件（默认复制）")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    if not src.is_dir():
        print(f"源文件夹不存在：{src}")
        return
    if not str(args.out).strip():
        print("请指定目标文件夹。")
        return
    if out.resolve() == src.resolve():
        print("目标文件夹不能与源文件夹相同。")
        return

    images = find_images(src, recursive=not args.no_recursive)
    if not images:
        print("源文件夹中没有找到图片。")
        return

    labels_dir = Path(args.labels).expanduser() if args.labels.strip() else None
    if labels_dir is not None and not labels_dir.is_dir():
        print(f"标签目录不存在（将不提取标签）：{labels_dir}")
        labels_dir = None

    out.mkdir(parents=True, exist_ok=True)
    used_targets = {}
    ok = 0
    skip = 0
    fail = 0
    label_ok = 0
    label_miss = 0
    action = shutil.move if args.move else shutil.copy2

    print(f"共找到 {len(images)} 张图片，目标文件夹：{out}")
    for i, img in enumerate(images, 1):
        if args.keep_structure:
            rel = img.relative_to(src)
            dst_dir = out / rel.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            key = str(dst_dir)
            used = used_targets.setdefault(key, set())
            name = resolve_conflict(dst_dir, img.name, args.conflict, used, img)
        else:
            dst_dir = out
            used = used_targets.setdefault(str(out), set())
            name = resolve_conflict(dst_dir, img.name, args.conflict, used, img)

        if name is None:
            skip += 1
            print(f"[{i}/{len(images)}] [跳过] {img.name}（同名已存在）")
            continue

        dst = dst_dir / name
        try:
            action(str(img), str(dst))
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(images)}] [失败] {img.name}：{exc}")
            continue

        ok += 1
        # 同步标签
        if args.with_labels:
            lbl = find_label_for(img, labels_dir)
            if lbl is not None:
                lbl_dst = dst.with_suffix(lbl.suffix)
                try:
                    shutil.copy2(str(lbl), str(lbl_dst))
                    label_ok += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"        [标签失败] {lbl.name}：{exc}")
            else:
                label_miss += 1

        extra = ""
        if args.keep_structure:
            extra = f"  ←  {img.parent.relative_to(src)}"
        print(f"[{i}/{len(images)}] {img.name} → {name}{extra}")

    summary = f"\n完成：提取 {ok} 张，跳过 {skip} 张，失败 {fail} 张"
    if args.with_labels:
        summary += f"，标签 {label_ok} 个，缺标签 {label_miss} 张"
    print(summary + "\n目标文件夹：" + str(out))


if __name__ == "__main__":
    main()
