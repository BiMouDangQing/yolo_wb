"""
标签映射命令行工具：把 YOLO 标签文件里指定类别 A 的框改成类别 B。

与前端模块 `QT/modules/label_remap.py` 对应，逻辑保持一致。

A / B 既可以是「类别名称」（自动读 labels 目录下的 classes.txt 解析成 ID），
也可以是「数字 ID」。例如把类别名 0-apple 的框全部改成 0：

用法:
    python tools/label_remap.py -i ./labels --src 0-apple --dst 0            # 原地改
    python tools/label_remap.py -i ./labels -o ./labels_new --src 0-apple --dst 0
"""

import argparse
from pathlib import Path


def load_class_names(labels_dir):
    """读取 labels 目录下的 classes.txt，返回类别名列表；不存在返回 []。"""
    classes_txt = Path(labels_dir) / "classes.txt"
    if classes_txt.is_file():
        try:
            return [
                line.strip()
                for line in classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
        except OSError:
            return []
    return []


def resolve_cls(value, class_names):
    """把输入解析成 class_id：纯数字直接当 ID，否则查类别名列表。失败返回 None。"""
    value = (value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value in class_names:
        return class_names.index(value)
    return None


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
    parser = argparse.ArgumentParser(description="把类别 A 的框改成类别 B（名称或 ID）")
    parser.add_argument("--labels", "-i", required=True, help="labels 目录")
    parser.add_argument("--out", "-o", default="", help="输出目录（不指定则原地修改）")
    parser.add_argument("--src", required=True, help="源类别（名称或 ID，如 0-apple 或 1）")
    parser.add_argument("--dst", required=True, help="目标类别（名称或 ID，如 0）")
    args = parser.parse_args()

    labels = Path(args.labels).expanduser()
    if not labels.is_dir():
        raise SystemExit(f"labels 目录不存在: {labels}")

    class_names = load_class_names(labels)
    src_cls = resolve_cls(args.src, class_names)
    dst_cls = resolve_cls(args.dst, class_names)
    if src_cls is None:
        raise SystemExit(f"无法解析源类别：{args.src!r}（可填数字 ID 或 classes.txt 里的类别名）")
    if dst_cls is None:
        raise SystemExit(f"无法解析目标类别：{args.dst!r}（可填数字 ID 或 classes.txt 里的类别名）")

    if class_names:
        print(f"类别表：{class_names}")

    files = sorted(p for p in labels.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".txt"
                   and p.name != "classes.txt")
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
            new_line = remap_line(line, src_cls, dst_cls)
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
        print(f"{src.name}: {changed} 框 {src_cls}->{dst_cls}")

    print(f"\n完成: 处理 {file_count} 个文件，改写 {box_count} 个框。")


if __name__ == "__main__":
    main()

