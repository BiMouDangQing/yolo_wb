"""
类别重命名命令行工具：把 YOLO 数据集里的类别名称 A 改成新名称 B。

与前端模块 `QT/modules/label_rename.py` 对应，逻辑保持一致。

只改 labels 目录下 classes.txt 里的类别名；框的 class_id 保持不变（A 在
classes.txt 中的位置不变），因此「名称 → id」的绑定关系自动正确更新。

用法:
    python tools/label_rename.py -i ./labels --src 0-apple --dst apple
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


def main() -> None:
    parser = argparse.ArgumentParser(description="把类别名称 A 改成 B")
    parser.add_argument("--labels", "-i", required=True, help="labels 目录（含 classes.txt）")
    parser.add_argument("--src", required=True, help="源类别名称 A（如 0-apple）")
    parser.add_argument("--dst", required=True, help="新类别名称 B（如 apple）")
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
        raise SystemExit(f"类别名 {src!r} 不存在于 classes.txt：{names}")
    if src == dst:
        raise SystemExit("源名称和目标名称相同，无需改动。")
    if dst in names:
        raise SystemExit(f"目标名称 {dst!r} 已存在，无法重命名（会造成类别冲突）。")

    idx = names.index(src)
    names[idx] = dst
    classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"类别 {src} (class_id={idx}) -> {dst}，class_id 保持不变。")
    print(f"已更新：{classes_txt}")
    print(f"新类别列表：{names}")


if __name__ == "__main__":
    main()
