"""标注质检模块：检测 YOLO 标签文件中的格式与几何硬伤。

检测项（第 1 层自动质检）：
- 格式错误：字段数不足 / 非数字；
- 越界框：cx / cy / w / h 超出 0~1 范围；
- 零/负宽高：w <= 0 或 h <= 0；
- 极小框：w 或 h 小于阈值（默认 0.01）；
- 巨框：w 或 h 大于阈值（默认 0.9）；
- 空标签：txt 为空或没有任何有效框；
- 类别 ID 越界：class < 0 或 class >= nc。

复用 analysis 的模式：QThread 后台处理，信号回传日志、进度与统计结果，
并生成异常明细 CSV。
"""

import csv
from pathlib import Path

from config import load as load_config, save as save_config
from qt_binding import QtCore, QtGui, QtWidgets, Signal

# 检测项：key -> 显示名
CHECK_ITEMS = [
    ("format", "格式错误"),
    ("out_of_range", "越界框"),
    ("zero_wh", "零/负宽高"),
    ("small", "极小框"),
    ("huge", "巨框"),
    ("empty", "空标签"),
    ("class_id", "类别ID越界"),
]


def check_label_file(path, min_wh, max_wh, nc, checks):
    """检测单个标签文件。

    返回 (issues, box_count)：
    - issues：异常列表，每项为 {"line": int, "type": str, "detail": str}；
    - box_count：有效框数量（用于空标签判断）。
    """
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
                issues.append({
                    "line": idx, "type": "格式错误",
                    "detail": f"字段数不足（{len(parts)} < 5）：{line}",
                })
            continue

        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            if checks.get("format"):
                issues.append({
                    "line": idx, "type": "格式错误",
                    "detail": f"无法解析为数字：{line}",
                })
            continue

        box_count += 1

        # 类别 ID 越界
        if nc > 0 and checks.get("class_id") and (cls < 0 or cls >= nc):
            issues.append({
                "line": idx, "type": "类别ID越界",
                "detail": f"class={cls}（有效范围 0~{nc - 1}）",
            })

        # 越界框
        if checks.get("out_of_range"):
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                issues.append({
                    "line": idx, "type": "越界框",
                    "detail": f"cx={cx}, cy={cy}, w={w}, h={h}",
                })
                continue

        # 零/负宽高
        if checks.get("zero_wh") and (w <= 0 or h <= 0):
            issues.append({
                "line": idx, "type": "零/负宽高",
                "detail": f"w={w}, h={h}",
            })
            continue

        # 极小框
        if checks.get("small") and (w < min_wh or h < min_wh):
            issues.append({
                "line": idx, "type": "极小框",
                "detail": f"w={w}, h={h}（阈值 {min_wh}）",
            })

        # 巨框
        if checks.get("huge") and (w > max_wh or h > max_wh):
            issues.append({
                "line": idx, "type": "巨框",
                "detail": f"w={w}, h={h}（阈值 {max_wh}）",
            })

    # 空标签
    if checks.get("empty") and box_count == 0:
        issues.append({"line": 0, "type": "空标签", "detail": "无有效标注框"})

    return issues, box_count


class LabelCheckWorker(QtCore.QThread):
    """后台执行标注质检，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)          # (当前进度, 总数)
    stats = Signal(object)               # 统计结果 dict
    finished = Signal(int, int, int)     # (文件数, 异常文件数, 异常总数)

    def __init__(self, labels_dir, min_wh, max_wh, nc, checks, csv_path, parent=None):
        super().__init__(parent)
        self.labels_dir = labels_dir
        self.min_wh = min_wh
        self.max_wh = max_wh
        self.nc = nc
        self.checks = checks              # {"format": bool, ...}
        self.csv_path = csv_path

    def _resolve_nc(self, labels_dir, txt_files):
        """确定类别数：优先用户输入，其次 classes.txt，最后最大 class_id + 1。"""
        if self.nc > 0:
            return self.nc
        classes_txt = labels_dir / "classes.txt"
        if classes_txt.is_file():
            try:
                names = [
                    line.strip()
                    for line in classes_txt.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    if line.strip()
                ]
                if names:
                    return len(names)
            except OSError:
                pass
        # 扫描所有 txt 找最大 class_id
        max_cls = -1
        for path in txt_files:
            try:
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        max_cls = max(max_cls, int(parts[0]))
                    except ValueError:
                        continue
            except OSError:
                pass
        return max_cls + 1

    def run(self):
        labels_dir = Path(self.labels_dir).expanduser()
        if not labels_dir.is_dir():
            self.log.emit(f"labels 目录不存在：{labels_dir}")
            self.finished.emit(0, 0, 0)
            return

        txt_files = sorted(
            p for p in labels_dir.rglob("*")
            if p.is_file() and p.suffix.upper() == ".TXT"
        )
        if not txt_files:
            self.log.emit("labels 目录中没有找到 .txt 标签文件。")
            self.finished.emit(0, 0, 0)
            return

        nc = self._resolve_nc(labels_dir, txt_files)
        self.log.emit(f"共 {len(txt_files)} 个标签文件，类别数 nc={nc}。")

        issue_count_by_type = {}
        issue_files = set()
        all_issues = []

        for i, path in enumerate(txt_files, 1):
            self.progress.emit(i, len(txt_files))
            issues, _box_count = check_label_file(
                path, self.min_wh, self.max_wh, nc, self.checks
            )
            if issues:
                issue_files.add(path)
                for issue in issues:
                    issue_count_by_type[issue["type"]] = (
                        issue_count_by_type.get(issue["type"], 0) + 1
                    )
                    all_issues.append({
                        "file": path.name,
                        "line": issue["line"],
                        "type": issue["type"],
                        "detail": issue["detail"],
                    })
                    self.log.emit(
                        f"[{issue['type']}] {path.name}"
                        + (f" 第 {issue['line']} 行" if issue["line"] else "")
                        + f"：{issue['detail']}"
                    )

        # 生成 CSV
        csv_path = None
        if self.csv_path:
            csv_path = Path(self.csv_path).expanduser()
            if csv_path.suffix.lower() != ".csv":
                csv_path = csv_path / "label_issues.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["文件", "行号", "异常类型", "详情"])
                for issue in all_issues:
                    writer.writerow([
                        issue["file"], issue["line"], issue["type"], issue["detail"],
                    ])
            self.log.emit(f"已生成异常明细 CSV：{csv_path}")

        stats = {
            "total_files": len(txt_files),
            "issue_files": len(issue_files),
            "issue_count_by_type": issue_count_by_type,
            "csv_path": str(csv_path) if csv_path else "",
        }
        self.stats.emit(stats)

        total_issues = sum(issue_count_by_type.values())
        self.log.emit(
            f"\n完成：共 {len(txt_files)} 个标签文件，"
            f"发现异常 {total_issues} 处，涉及 {len(issue_files)} 个文件。"
        )
        self.finished.emit(len(txt_files), len(issue_files), total_issues)


class LabelCheckModule(QtWidgets.QWidget):
    MODULE_TITLE = "标注质检"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._checks = {}
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "检测 YOLO 标签（.txt）的格式与几何硬伤：格式错误、越界框、\n"
            "零/负宽高、极小框、巨框、空标签、类别 ID 越界，并生成异常明细 CSV。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # labels 目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("labels 目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("YOLO 标签文件夹（必填）")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 参数
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("类别数 nc:"))
        self.nc_spin = QtWidgets.QSpinBox()
        self.nc_spin.setRange(0, 10000)
        self.nc_spin.setValue(0)
        self.nc_spin.setToolTip("0 表示自动（读 classes.txt 或按最大 class_id+1）")
        param_row.addWidget(self.nc_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("极小框阈值:"))
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setRange(0.0, 1.0)
        self.min_spin.setSingleStep(0.005)
        self.min_spin.setDecimals(3)
        self.min_spin.setValue(0.01)
        param_row.addWidget(self.min_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("巨框阈值:"))
        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setRange(0.0, 1.0)
        self.max_spin.setSingleStep(0.05)
        self.max_spin.setDecimals(2)
        self.max_spin.setValue(0.9)
        param_row.addWidget(self.max_spin)
        param_row.addStretch()
        layout.addLayout(param_row)

        # 检测项
        check_box = QtWidgets.QGroupBox("检测项")
        check_layout = QtWidgets.QGridLayout(check_box)
        for i, (key, label) in enumerate(CHECK_ITEMS):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(True)
            self._checks[key] = check
            check_layout.addWidget(check, i // 3, i % 3)
        layout.addWidget(check_box)

        # CSV 输出
        csv_row = QtWidgets.QHBoxLayout()
        csv_row.addWidget(QtWidgets.QLabel("CSV 输出:"))
        self.csv_edit = QtWidgets.QLineEdit()
        self.csv_edit.setPlaceholderText("留空表示不生成 CSV")
        csv_row.addWidget(self.csv_edit, 1)
        csv_btn = QtWidgets.QPushButton("浏览")
        csv_btn.clicked.connect(self._pick_csv)
        csv_row.addWidget(csv_btn)
        layout.addLayout(csv_row)

        # 统计表格
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["异常类型", "数量"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始检测")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._start)
        layout.addWidget(self.start_btn)

        # 日志
        layout.addWidget(QtWidgets.QLabel("日志:"))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择 CSV 保存位置", "", "CSV 文件 (*.csv)"
        )
        if path:
            self.csv_edit.setText(path)

    def _restore_config(self):
        """恢复上次保存的路径与参数。"""
        cfg = load_config("label_check")
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.csv_edit.setText(cfg.get("csv_path", ""))
        self.nc_spin.setValue(int(cfg.get("nc", 0)))
        self.min_spin.setValue(float(cfg.get("min_wh", 0.01)))
        self.max_spin.setValue(float(cfg.get("max_wh", 0.9)))
        checks = cfg.get("checks", {})
        for key, check in self._checks.items():
            check.setChecked(bool(checks.get(key, True)))

    def _persist_config(self):
        """保存当前路径与参数。"""
        save_config("label_check", {
            "labels_dir": self.labels_edit.text().strip(),
            "csv_path": self.csv_edit.text().strip(),
            "nc": self.nc_spin.value(),
            "min_wh": self.min_spin.value(),
            "max_wh": self.max_spin.value(),
            "checks": {k: c.isChecked() for k, c in self._checks.items()},
        })

    def _start(self):
        labels_dir = self.labels_edit.text().strip()
        if not labels_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 labels 目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = LabelCheckWorker(
            labels_dir=labels_dir,
            min_wh=self.min_spin.value(),
            max_wh=self.max_spin.value(),
            nc=self.nc_spin.value(),
            checks={k: c.isChecked() for k, c in self._checks.items()},
            csv_path=self.csv_edit.text().strip(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.stats.connect(self._on_stats)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_stats(self, stats):
        self.table.setRowCount(0)
        for typ, count in sorted(stats["issue_count_by_type"].items()):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(typ))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(count)))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _on_finished(self, total_files, issue_files, total_issues):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成",
            f"检测完成：共 {total_files} 个标签文件\n"
            f"发现异常 {total_issues} 处，涉及 {issue_files} 个文件。",
        )
