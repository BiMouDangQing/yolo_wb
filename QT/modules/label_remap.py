"""标签映射模块：把 YOLO 标签文件里指定类别 A 的框，改成类别 B。

A / B 既可以是「类别名称」（自动读 labels 目录下的 classes.txt 解析成 ID），
也可以是「数字 ID」。例如把类别名 0-apple 的框全部改成 0。
只改匹配到源类别的框，其余行原样保留。
"""

from pathlib import Path

from config import load as load_config, save as save_config
from logs import write_log
from modules._history import HistoryBar
from qt_binding import QtCore, QtGui, QtWidgets, Signal


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
    """把一行标签中类别等于 src_cls 的框改成 dst_cls，返回新行；否则原样返回。"""
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


class LabelRemapWorker(QtCore.QThread):
    """后台执行标签映射，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    finished = Signal(int, int)      # (文件数, 修改框数)

    def __init__(self, labels_dir, output_dir, inplace, src, dst, parent=None):
        super().__init__(parent)
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.inplace = inplace
        self.src = src          # 源类别（名称或 ID）
        self.dst = dst          # 目标类别（名称或 ID）

    def run(self):
        labels = Path(self.labels_dir).expanduser()
        if not labels.is_dir():
            self.log.emit(f"labels 目录不存在：{labels}")
            self.finished.emit(0, 0)
            return

        class_names = load_class_names(labels)
        src_cls = resolve_cls(self.src, class_names)
        dst_cls = resolve_cls(self.dst, class_names)
        if src_cls is None:
            self.log.emit(f"无法解析源类别：{self.src!r}（可填数字 ID 或 classes.txt 里的类别名）")
            self.finished.emit(0, 0)
            return
        if dst_cls is None:
            self.log.emit(f"无法解析目标类别：{self.dst!r}（可填数字 ID 或 classes.txt 里的类别名）")
            self.finished.emit(0, 0)
            return
        if class_names:
            self.log.emit(f"类别表：{class_names}")

        files = sorted(p for p in labels.rglob("*")
                       if p.is_file() and p.suffix.lower() == ".txt"
                       and p.name != "classes.txt")
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
                new_line = remap_line(line, src_cls, dst_cls)
                new_lines.append(new_line)
                old_parts = line.split()
                new_parts = new_line.split()
                if old_parts and new_parts and old_parts[0] != new_parts[0]:
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
            self.log.emit(f"[{i + 1}/{total}] {src.name}：{changed} 框 {src_cls}→{dst_cls}")

        self.log.emit(f"\n完成：处理 {file_count} 个文件，共改写 {box_count} 个框。")
        self.finished.emit(file_count, box_count)


class LabelRemapModule(QtWidgets.QWidget):
    MODULE_TITLE = "标签映射"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把 YOLO 标签文件里指定类别 A 的框改成类别 B。\n"
            "A / B 可填「类别名称」（如 0-apple）或「数字 ID」（如 0），\n"
            "会自动读 labels 目录下的 classes.txt 解析；只改匹配到 A 的框，其余不变。"
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

        # 源类别 -> 目标类别（可输入名称或 ID）
        cls_row = QtWidgets.QHBoxLayout()
        cls_row.addWidget(QtWidgets.QLabel("源类别 A:"))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setEditable(True)
        self.src_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.src_combo.setMinimumWidth(160)
        self.src_combo.setToolTip("类别名称（如 0-apple）或数字 ID（如 1）")
        cls_row.addWidget(self.src_combo)
        cls_row.addSpacing(16)
        cls_row.addWidget(QtWidgets.QLabel("目标类别 B:"))
        self.dst_combo = QtWidgets.QComboBox()
        self.dst_combo.setEditable(True)
        self.dst_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.dst_combo.setMinimumWidth(160)
        self.dst_combo.setToolTip("类别名称或数字 ID（如 0）")
        cls_row.addWidget(self.dst_combo)
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
        self.history_bar = HistoryBar("label_remap", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始映射")
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
            self._load_class_names(path)

    def _load_class_names(self, labels_dir):
        """把 labels 目录下 classes.txt 的类别名加载到下拉框（不覆盖用户已输入的文字）。"""
        names = load_class_names(labels_dir)
        if not names:
            return
        for combo in (self.src_combo, self.dst_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            # 保留用户已经输入的文字
            combo.setEditText(current if current else "")
            combo.blockSignals(False)

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
        self._apply_config(load_config("label_remap"))

    def _apply_config(self, cfg):
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.output_edit.setText(cfg.get("output_dir", ""))
        self.src_combo.setEditText(str(cfg.get("src", "1")))
        self.dst_combo.setEditText(str(cfg.get("dst", "0")))
        mode = cfg.get("mode", "inplace")
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        labels_dir = cfg.get("labels_dir", "")
        if labels_dir:
            self._load_class_names(labels_dir)

    def _persist_config(self):
        save_config("label_remap", {
            "labels_dir": self.labels_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "src": self.src_combo.currentText().strip(),
            "dst": self.dst_combo.currentText().strip(),
            "mode": self.mode_combo.currentData(),
        })

    def _start(self):
        labels_dir = self.labels_edit.text().strip()
        if not labels_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 labels 目录。")
            return
        src = self.src_combo.currentText().strip()
        dst = self.dst_combo.currentText().strip()
        if not src or not dst:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写源类别 A 和目标类别 B（名称或 ID）。")
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

        self._worker = LabelRemapWorker(
            labels_dir=labels_dir,
            output_dir=output_dir,
            inplace=inplace,
            src=src,
            dst=dst,
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
