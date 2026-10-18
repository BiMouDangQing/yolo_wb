"""标签统一模块：把 YOLO 标签文件里所有框的类别 ID 统一改成指定类别。

例如：多类别（0/1/2）标注的数据集，一键全部改成类别 0，用于单类别检测。

只改每行第一列的类别 ID，其余字段（cx/cy/w/h 及可能的额外坐标）保持不变。
"""

from pathlib import Path

from config import load as load_config, save as save_config
from logs import write_log
from modules._history import HistoryBar
from qt_binding import QtCore, QtGui, QtWidgets, Signal


def remap_line(line, target_cls):
    """把一行标签的类别 ID 改成 target_cls，返回新行；无法解析时返回 None。"""
    line = line.rstrip("\n")
    stripped = line.strip()
    if not stripped:
        return line
    parts = stripped.split()
    if not parts:
        return line
    parts[0] = str(target_cls)
    return " ".join(parts)


class LabelUnifyWorker(QtCore.QThread):
    """后台执行标签统一，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    finished = Signal(int, int)      # (文件数, 修改框数)

    def __init__(self, labels_dir, output_dir, inplace, target_cls, parent=None):
        super().__init__(parent)
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.inplace = inplace
        self.target_cls = target_cls

    def run(self):
        labels = Path(self.labels_dir).expanduser()
        if not labels.is_dir():
            self.log.emit(f"labels 目录不存在：{labels}")
            self.finished.emit(0, 0)
            return

        files = sorted(p for p in labels.rglob("*")
                       if p.is_file() and p.suffix.lower() == ".txt")
        if not files:
            self.log.emit("labels 目录中没有找到 .txt 标签文件。")
            self.finished.emit(0, 0)
            return

        out = None
        if not self.inplace:
            if not str(self.output_dir).strip():
                self.log.emit("输出模式下必须指定输出目录。")
                self.finished.emit(0, 0)
                return
            out = Path(self.output_dir).expanduser()
            out.mkdir(parents=True, exist_ok=True)

        total = len(files)
        file_count = 0
        box_count = 0
        for i, src in enumerate(files):
            self.progress.emit(i + 1, total)
            try:
                raw = src.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                self.log.emit(f"[失败] 无法读取 {src.name}：{exc}")
                continue

            lines = raw.splitlines()
            new_lines = []
            changed = 0
            for line in lines:
                new_line = remap_line(line, self.target_cls)
                new_lines.append(new_line)
                # 粗略统计：行内第一个字段被改写即算一个框
                if new_line.split() and line.split():
                    if line.split()[0] != new_line.split()[0]:
                        changed += 1

            if self.inplace:
                dst = src
            else:
                dst = out / src.name
            try:
                dst.write_text("\n".join(new_lines) + ("\n" if raw.endswith("\n") else ""),
                               encoding="utf-8")
            except OSError as exc:
                self.log.emit(f"[失败] 无法写入 {dst.name}：{exc}")
                continue

            file_count += 1
            box_count += changed
            self.log.emit(f"[{i + 1}/{total}] {src.name} → 类别 {self.target_cls}（{changed} 框）")

        self.log.emit(f"\n完成：处理 {file_count} 个文件，共改写 {box_count} 个框。")
        self.finished.emit(file_count, box_count)


class LabelUnifyModule(QtWidgets.QWidget):
    MODULE_TITLE = "标签统一"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把 YOLO 标签文件里所有框的类别 ID 统一改成指定类别（做单类别检测）。\n"
            "只改每行第一列的类别 ID，框坐标保持不变。"
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

        # 目标类别
        cls_row = QtWidgets.QHBoxLayout()
        cls_row.addWidget(QtWidgets.QLabel("目标类别 ID:"))
        self.target_spin = QtWidgets.QSpinBox()
        self.target_spin.setRange(0, 9999)
        self.target_spin.setValue(0)
        cls_row.addWidget(self.target_spin)
        cls_row.addStretch()
        layout.addLayout(cls_row)

        # 输出方式
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("输出方式:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("原地修改（覆盖原文件）", "inplace")
        self.mode_combo.addItem("输出到新目录", "copy")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # 输出目录（仅输出模式）
        self.out_row = QtWidgets.QHBoxLayout()
        self.out_row.addWidget(QtWidgets.QLabel("输出目录:"))
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("改写后的标签保存位置（输出模式必填）")
        self.out_row.addWidget(self.output_edit, 1)
        self.out_btn = QtWidgets.QPushButton("浏览")
        self.out_btn.clicked.connect(self._pick_output)
        self.out_row.addWidget(self.out_btn)
        layout.addLayout(self.out_row)

        # 历史配置恢复
        self.history_bar = HistoryBar("label_unify", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始统一")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._start)
        layout.addWidget(self.start_btn)

        # 日志
        layout.addWidget(QtWidgets.QLabel("日志:"))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

        self._on_mode_changed()

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def _on_mode_changed(self):
        inplace = self.mode_combo.currentData() == "inplace"
        self.out_row.itemAt(0).widget().setEnabled(not inplace)
        self.output_edit.setEnabled(not inplace)
        for i in range(1, self.out_row.count()):
            w = self.out_row.itemAt(i).widget()
            if w is not None:
                w.setEnabled(not inplace)

    def _restore_config(self):
        self._apply_config(load_config("label_unify"))

    def _apply_config(self, cfg):
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.output_edit.setText(cfg.get("output_dir", ""))
        self.target_spin.setValue(int(cfg.get("target_cls", 0)))
        mode = cfg.get("mode", "inplace")
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def _persist_config(self):
        save_config("label_unify", {
            "labels_dir": self.labels_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "target_cls": self.target_spin.value(),
            "mode": self.mode_combo.currentData(),
        })

    def _start(self):
        labels_dir = self.labels_edit.text().strip()
        if not labels_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 labels 目录。")
            return

        inplace = self.mode_combo.currentData() == "inplace"
        output_dir = self.output_edit.text().strip()
        if not inplace and not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "输出模式下请选择输出目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = LabelUnifyWorker(
            labels_dir=labels_dir,
            output_dir=output_dir,
            inplace=inplace,
            target_cls=self.target_spin.value(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)
        write_log(self.MODULE_TITLE, text)

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_finished(self, count, boxes):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"处理完成：{count} 个文件，改写 {boxes} 个框。"
        )
