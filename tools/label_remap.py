"""
标签映射命令行工具：把 YOLO 标签文件里指定类别 A 的框改成类别 B。

与前端模块 `QT/modules/label_remap.py` 对应，逻辑保持一致。

用法:
    python tools/label_remap.py -i ./labels --src 1 --dst 0            # 原地改
    python tools/label_remap.py -i ./labels -o ./labels_new --src 1 --dst 0
"""

import argparse
from pathlib import Path


def remap_line(line, src_cls, dst_cls):
    line = line.rstrip("\n")
    stripped = line.strip()
    if not stripped:
        return line
    parts = stripped.split()
    if not parts:
        return line
    try:
        if int(parts[0]) == src_cls:
            parts[0] = str(dst_cls)
    except ValueError:
        return line
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="把类别 A 的框改成类别 B")
    parser.add_argument("--labels", "-i", required=True, help="labels 目录")
    parser.add_argument("--out", "-o", default="", help="输出目录（不指定则原地修改）")
    parser.add_argument("--src", type=int, required=True, help="源类别 A")
    parser.add_argument("--dst", type=int, required=True, help="目标类别 B")
    args = parser.parse_args()

    labels = Path(args.labels).expanduser()
    if not labels.is_dir():
        raise SystemExit(f"labels 目录不存在: {labels}")

    files = sorted(p for p in labels.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".txt")
    if not files:
        raise SystemExit("labels 目录中没有找到 .txt 标签文件。")

    out = Path(args.out).expanduser() if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    file_count = 0
    box_count = 0
    for src in files:
        raw = src.read_text(encoding="utf-8", errors="ignore")
        lines = raw.splitlines()
        new_lines = []
        changed = 0
        for line in lines:
            new_line = remap_line(line, args.src, args.dst)
            new_lines.append(new_line)
            old_parts = line.split()
            new_parts = new_line.split()
            if old_parts and new_parts and old_parts[0] != new_parts[0]:
                changed += 1

        dst = src if out is None else out / src.name
        dst.write_text("\n".join(new_lines) + ("\n" if raw.endswith("\n") else ""),
                       encoding="utf-8")
        file_count += 1
        box_count += changed
        print(f"{src.name}: {changed} 框 {args.src}->{args.dst}")

    print(f"\n完成: 处理 {file_count} 个文件，改写 {box_count} 个框。")


if __name__ == "__main__":
    main()
