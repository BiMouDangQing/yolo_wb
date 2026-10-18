"""类别合并模块：把 YOLO 数据集里的类别 C 合并到类别 B，id 正确重排。

合并过程：
1. 把类别 C 的框的 class_id 改成 B 的 id；
2. 从 classes.txt 删除 C 那一行；
3. 原本在 C 之后的所有类别 id 前移一位，并同步更新所有框的 txt。
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


def remap_id(cls, src_id, dst_id):
    """把单个框的 class_id 按合并规则重排。"""
    if cls == src_id:
        return dst_id - 1 if dst_id > src_id else dst_id
    if cls > src_id:
        return cls - 1
    return cls


class LabelMergeWorker(QtCore.QThread):
    """后台执行类别合并，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    finished = Signal(int, int)      # (文件数, 调整框数)

    def __init__(self, labels_dir, src_name, dst_name, parent=None):
        super().__init__(parent)
        self.labels_dir = labels_dir
        self.src_name = src_name
        self.dst_name = dst_name

    def run(self):
        labels = Path(self.labels_dir).expanduser()
        classes_txt = labels / "classes.txt"
        if not classes_txt.is_file():
            self.log.emit(f"找不到 classes.txt：{classes_txt}")
            self.finished.emit(0, 0)
            return

        names = load_class_names(labels)
        if not names:
            self.log.emit("classes.txt 为空。")
            self.finished.emit(0, 0)
            return

        src = self.src_name.strip()
        dst = self.dst_name.strip()
        if src not in names:
            self.log.emit(f"源类别 {src!r} 不存在于 classes.txt：{names}")
            self.finished.emit(0, 0)
            return
        if dst not in names:
            self.log.emit(f"目标类别 {dst!r} 不存在于 classes.txt：{names}")
            self.finished.emit(0, 0)
            return
        if src == dst:
            self.log.emit("源类别和目标类别相同，无需合并。")
            self.finished.emit(0, 0)
            return

        src_id = names.index(src)
        dst_id = names.index(dst)
        self.log.emit(f"合并 {src} (id={src_id}) → {dst} (id={dst_id})")

        files = sorted(p for p in labels.rglob("*")
                       if p.is_file() and p.suffix.lower() == ".txt"
                       and p.name != "classes.txt")
        if not files:
            self.log.emit("labels 目录中没有找到框的 .txt 文件。")
            self.finished.emit(0, 0)
            return

        total = len(files)
        file_count = 0
        box_count = 0
        for i, fp in enumerate(files):
            self.progress.emit(i + 1, total)
            try:
                raw = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                self.log.emit(f"[失败] 无法读取 {fp.name}：{exc}")
                continue

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

            try:
                fp.write_text("\n".join(new_lines) + ("\n" if raw.endswith("\n") else ""),
                              encoding="utf-8")
            except OSError as exc:
                self.log.emit(f"[失败] 无法写入 {fp.name}：{exc}")
                continue

            file_count += 1
            box_count += changed
            self.log.emit(f"[{i + 1}/{total}] {fp.name}：调整 {changed} 个框")

        # 删除源类别，写回 classes.txt
        names.pop(src_id)
        classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")

        self.log.emit(f"\n完成：处理 {file_count} 个文件，调整 {box_count} 个框的 id。")
        self.log.emit(f"类别 {src} (id={src_id}) 已合并到 {dst}，新类别列表：{names}")
        self.finished.emit(file_count, box_count)


class LabelMergeModule(QtWidgets.QWidget):
    MODULE_TITLE = "类别合并"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把类别 C 合并到类别 B（如 1-LightBruise → apple）。\n"
            "会把 C 的框的 class_id 改成 B 的 id，并从 classes.txt 删除 C，\n"
            "C 之后所有类别的 id 自动前移，框的 txt 同步更新（id 完整重新绑定）。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # labels 目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("labels 目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("YOLO 标签文件夹（含 classes.txt，必填）")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 源类别 C
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(QtWidgets.QLabel("源类别 C:"))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setEditable(True)
        self.src_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.src_combo.setMinimumWidth(200)
        self.src_combo.setToolTip("要合并掉的类别，如 1-LightBruise")
        src_row.addWidget(self.src_combo)
        src_row.addStretch()
        layout.addLayout(src_row)

        # 目标类别 B
        dst_row = QtWidgets.QHBoxLayout()
        dst_row.addWidget(QtWidgets.QLabel("目标类别 B:")
        )
        self.dst_combo = QtWidgets.QComboBox()
        self.dst_combo.setEditable(True)
        self.dst_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.dst_combo.setMinimumWidth(200)
        self.dst_combo.setToolTip("合并到的目标类别，如 apple")
        dst_row.addWidget(self.dst_combo)
        dst_row.addStretch()
        layout.addLayout(dst_row)

        # 历史配置恢复
        self.history_bar = HistoryBar("label_merge", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始合并")
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

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)
            self._load_class_names(path)

    def _load_class_names(self, labels_dir):
        names = load_class_names(labels_dir)
        if not names:
            return
        for combo in (self.src_combo, self.dst_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setEditText(current if current else "")
            combo.blockSignals(False)

    def _restore_config(self):
        self._apply_config(load_config("label_merge"))

    def _apply_config(self, cfg):
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.src_combo.setEditText(str(cfg.get("src", "")))
        self.dst_combo.setEditText(str(cfg.get("dst", "")))
        labels_dir = cfg.get("labels_dir", "")
        if labels_dir:
            self._load_class_names(labels_dir)

    def _persist_config(self):
        save_config("label_merge", {
            "labels_dir": self.labels_edit.text().strip(),
            "src": self.src_combo.currentText().strip(),
            "dst": self.dst_combo.currentText().strip(),
        })

    def _start(self):
        labels_dir = self.labels_edit.text().strip()
        if not labels_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 labels 目录。")
            return
        src = self.src_combo.currentText().strip()
        dst = self.dst_combo.currentText().strip()
        if not src or not dst:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写源类别 C 和目标类别 B。")
            return

        self._persist_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = LabelMergeWorker(
            labels_dir=labels_dir,
            src_name=src,
            dst_name=dst,
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
            self, "完成", f"合并完成：处理 {count} 个文件，调整 {boxes} 个框的 id。"
        )
