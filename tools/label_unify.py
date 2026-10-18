"""
标签统一命令行工具：把 YOLO 标签文件里所有框的类别 ID 统一改成指定类别。

与前端模块 `QT/modules/label_unify.py` 对应，逻辑保持一致。

用法:
    python tools/label_unify.py -i ./labels --target 0            # 原地改
    python tools/label_unify.py -i ./labels -o ./labels_new --target 0
"""

import argparse
from pathlib import Path


def remap_line(line, target_cls):
    line = line.rstrip("\n")
    stripped = line.strip()
    if not stripped:
        return line
    parts = stripped.split()
    if not parts:
        return line
    parts[0] = str(target_cls)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="把所有框的类别 ID 统一改成指定类别")
    parser.add_argument("--labels", "-i", required=True, help="labels 目录")
    parser.add_argument("--out", "-o", default="", help="输出目录（不指定则原地修改）")
    parser.add_argument("--target", type=int, default=0, help="目标类别 ID，默认 0")
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
            new_line = remap_line(line, args.target)
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
        print(f"{src.name} -> 类别 {args.target}（{changed} 框）")

    print(f"\n完成: 处理 {file_count} 个文件，改写 {box_count} 个框。")


if __name__ == "__main__":
    main()
