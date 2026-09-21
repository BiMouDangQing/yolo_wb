"""
标注质检命令行工具：检测 YOLO 标签文件的格式与几何硬伤。

与前端模块 `QT/modules/label_check.py` 对应，逻辑保持一致。
检测项：格式错误、越界框、零/负宽高、极小框、巨框、空标签、类别 ID 越界。

用法:
    python tools/label_check.py -i ./labels -o ./label_issues.csv
    python tools/label_check.py -i ./labels --nc 2 --min-wh 0.01 --max-wh 0.9
"""

import argparse
import csv
from pathlib import Path


def check_label_file(path, min_wh, max_wh, nc, checks):
    """检测单个标签文件，返回 (issues, box_count)。"""
    issues = []
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return [{"line": 0, "type": "读取失败", "detail": str(exc)}], 0

    box_count = 0
    for idx, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            if checks.get("format"):
                issues.append({"line": idx, "type": "格式错误",
                               "detail": f"字段数不足：{line}"})
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            if checks.get("format"):
                issues.append({"line": idx, "type": "格式错误",
                               "detail": f"无法解析为数字：{line}"})
            continue

        box_count += 1

        if nc > 0 and checks.get("class_id") and (cls < 0 or cls >= nc):
            issues.append({"line": idx, "type": "类别ID越界",
                           "detail": f"class={cls}（有效范围 0~{nc - 1}）"})

        if checks.get("out_of_range"):
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                issues.append({"line": idx, "type": "越界框",
                               "detail": f"cx={cx}, cy={cy}, w={w}, h={h}"})
                continue

        if checks.get("zero_wh") and (w <= 0 or h <= 0):
            issues.append({"line": idx, "type": "零/负宽高",
                           "detail": f"w={w}, h={h}"})
            continue

        if checks.get("small") and (w < min_wh or h < min_wh):
            issues.append({"line": idx, "type": "极小框",
                           "detail": f"w={w}, h={h}（阈值 {min_wh}）"})

        if checks.get("huge") and (w > max_wh or h > max_wh):
            issues.append({"line": idx, "type": "巨框",
                           "detail": f"w={w}, h={h}（阈值 {max_wh}）"})

    if checks.get("empty") and box_count == 0:
        issues.append({"line": 0, "type": "空标签", "detail": "无有效标注框"})

    return issues, box_count


def resolve_nc(labels_dir, txt_files, user_nc):
    if user_nc > 0:
        return user_nc
    classes_txt = labels_dir / "classes.txt"
    if classes_txt.is_file():
        names = [line.strip() for line in classes_txt.read_text(
            encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        if names:
            return len(names)
    max_cls = -1
    for path in txt_files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                max_cls = max(max_cls, int(parts[0]))
            except ValueError:
                continue
    return max_cls + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="检测 YOLO 标签文件的格式与几何硬伤")
    parser.add_argument("--input", "-i", required=True, help="labels 目录")
    parser.add_argument("--output", "-o", default="", help="CSV 异常明细输出路径（可选）")
    parser.add_argument("--nc", type=int, default=0, help="类别数（0=自动）")
    parser.add_argument("--min-wh", type=float, default=0.01, help="极小框阈值（默认 0.01）")
    parser.add_argument("--max-wh", type=float, default=0.9, help="巨框阈值（默认 0.9）")
    parser.add_argument("--no-format", action="store_true", help="关闭格式错误检测")
    parser.add_argument("--no-out-of-range", action="store_true", help="关闭越界框检测")
    parser.add_argument("--no-zero-wh", action="store_true", help="关闭零/负宽高检测")
    parser.add_argument("--no-small", action="store_true", help="关闭极小框检测")
    parser.add_argument("--no-huge", action="store_true", help="关闭巨框检测")
    parser.add_argument("--no-empty", action="store_true", help="关闭空标签检测")
    parser.add_argument("--no-class-id", action="store_true", help="关闭类别ID越界检测")
    args = parser.parse_args()

    labels_dir = Path(args.input).expanduser().resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"labels 目录不存在: {labels_dir}")

    txt_files = sorted(p for p in labels_dir.rglob("*")
                       if p.is_file() and p.suffix.upper() == ".TXT")
    if not txt_files:
        raise SystemExit("labels 目录中没有找到 .txt 文件。")

    checks = {
        "format": not args.no_format,
        "out_of_range": not args.no_out_of_range,
        "zero_wh": not args.no_zero_wh,
        "small": not args.no_small,
        "huge": not args.no_huge,
        "empty": not args.no_empty,
        "class_id": not args.no_class_id,
    }

    nc = resolve_nc(labels_dir, txt_files, args.nc)
    print(f"共 {len(txt_files)} 个标签文件，类别数 nc={nc}。")

    count_by_type = {}
    all_issues = []
    for path in txt_files:
        issues, _ = check_label_file(path, args.min_wh, args.max_wh, nc, checks)
        for issue in issues:
            count_by_type[issue["type"]] = count_by_type.get(issue["type"], 0) + 1
            all_issues.append({
                "file": path.name, "line": issue["line"],
                "type": issue["type"], "detail": issue["detail"],
            })
            print(f"[{issue['type']}] {path.name}"
                  + (f" 第 {issue['line']} 行" if issue["line"] else "")
                  + f"：{issue['detail']}")

    if args.output:
        csv_path = Path(args.output).expanduser().resolve()
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path / "label_issues.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["文件", "行号", "异常类型", "详情"])
            for issue in all_issues:
                writer.writerow([issue["file"], issue["line"],
                                 issue["type"], issue["detail"]])
        print(f"CSV 已生成: {csv_path}")

    total = sum(count_by_type.values())
    print(f"\n完成: 共 {len(txt_files)} 个文件，发现异常 {total} 处。")
    for typ, count in sorted(count_by_type.items()):
        print(f"  {typ}: {count}")


if __name__ == "__main__":
    main()
