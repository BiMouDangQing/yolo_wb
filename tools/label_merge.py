"""
类别合并命令行工具：把 YOLO 数据集里的类别 C 合并到类别 B。

与前端模块 `QT/modules/label_merge.py` 对应，逻辑保持一致。

合并过程会正确重排 class_id：
1. 把类别 C 的框的 class_id 改成 B 的 id；
2. 从 classes.txt 删除 C 那一行；
3. 原本在 C 之后的所有类别 id 前移一位，并同步更新所有框的 txt。

用法:
    python tools/label_merge.py -i ./labels --src C --dst B
    python tools/label_merge.py -i ./labels --src 1-LightBruise --dst apple
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


def remap_id(cls, src_id, dst_id):
    """把单个框的 class_id 按合并规则重排。

    删除 src_id 这一行后，所有 > src_id 的类别前移一位；
    源类别 src_id 归到目标类别 dst_id（dst 若在 src 之后也要前移一位）。
    """
    if cls == src_id:
        return dst_id - 1 if dst_id > src_id else dst_id
    if cls > src_id:
        return cls - 1
    return cls


def main() -> None:
    parser = argparse.ArgumentParser(description="把类别 C 合并到类别 B，id 重新绑定")
    parser.add_argument("--labels", "-i", required=True, help="labels 目录（含 classes.txt）")
    parser.add_argument("--src", required=True, help="要合并掉的源类别 C（名称）")
    parser.add_argument("--dst", required=True, help="合并目标类别 B（名称）")
    args = parser.parse_args()

    labels = Path(args.labels).expanduser()
    classes_txt = labels / "classes.txt"
    if not classes_txt.is_file():
        raise SystemExit(f"找不到 classes.txt: {classes_txt}")

    names = load_class_names(labels)
    if not names:
        raise SystemExit("classes.txt 为空。")

    src = args.src.strip()
    dst = args.dst.strip()
    if src not in names:
        raise SystemExit(f"源类别 {src!r} 不存在于 classes.txt：{names}")
    if dst not in names:
        raise SystemExit(f"目标类别 {dst!r} 不存在于 classes.txt：{names}")
    if src == dst:
        raise SystemExit("源类别和目标类别相同，无需合并。")

    src_id = names.index(src)
    dst_id = names.index(dst)

    files = sorted(p for p in labels.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".txt"
                   and p.name != "classes.txt")
    if not files:
        raise SystemExit("labels 目录中没有找到框的 .txt 文件。")

    total = len(files)
    file_count = 0
    box_count = 0
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        lines = raw.splitlines()
        new_lines = []
        changed = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line.rstrip("\n"))
                continue
            parts = stripped.split()
            try:
                cls = int(parts[0])
            except ValueError:
                new_lines.append(line.rstrip("\n"))
                continue
            new_cls = remap_id(cls, src_id, dst_id)
            if new_cls != cls:
                changed += 1
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))

        fp.write_text("\n".join(new_lines) + ("\n" if raw.endswith("\n") else ""),
                       encoding="utf-8")
        file_count += 1
        box_count += changed
        print(f"{fp.name}: 调整 {changed} 个框")

    # 删除源类别，写回 classes.txt
    names.pop(src_id)
    classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")

    print(f"\n完成: 处理 {file_count} 个文件，调整 {box_count} 个框的 id。")
    print(f"类别 {src} (id={src_id}) 已合并到 {dst} (id={dst_id})。")
    print(f"新类别列表：{names}")


if __name__ == "__main__":
    main()
