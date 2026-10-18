"""类别重命名模块：把 YOLO 数据集里的类别名称 A 改成新名称 B。

只改 labels 目录下 classes.txt 里的类别名；框的 class_id 保持不变（A 在
classes.txt 中的位置不变），因此「名称 → id」的绑定关系自动正确更新。
用于把带前缀的类别名（如 0-apple）规整成干净名称（如 apple）。
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


class LabelRenameWorker(QtCore.QThread):
    """后台执行类别重命名，避免阻塞界面。"""

    log = Signal(str)
    finished = Signal(bool)  # 是否成功

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
            self.finished.emit(False)
            return

        names = load_class_names(labels)
        if not names:
            self.log.emit("classes.txt 为空。")
            self.finished.emit(False)
            return

        src = self.src_name.strip()
        dst = self.dst_name.strip()
        if src not in names:
            self.log.emit(f"类别名 {src!r} 不存在于 classes.txt：{names}")
            self.finished.emit(False)
            return
        if src == dst:
            self.log.emit("源名称和目标名称相同，无需改动。")
            self.finished.emit(False)
            return
        if dst in names:
            self.log.emit(f"目标名称 {dst!r} 已存在，无法重命名（会造成类别冲突）。")
            self.finished.emit(False)
            return

        idx = names.index(src)
        names[idx] = dst
        classes_txt.write_text("\n".join(names) + "\n", encoding="utf-8")
        self.log.emit(f"类别 {src} (class_id={idx}) -> {dst}，class_id 保持不变。")
        self.log.emit(f"已更新：{classes_txt}")
        self.log.emit(f"新类别列表：{names}")
        self.finished.emit(True)


class LabelRenameModule(QtWidgets.QWidget):
    MODULE_TITLE = "类别重命名"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把 YOLO 数据集里某个类别名称 A 改成新名称 B（如 0-apple → apple）。\n"
            "只改 classes.txt 里的类别名，框的 class_id 保持不变（类别位置不变），\n"
            "「名称 → id」的绑定关系会自动正确更新。"
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

        # 源类别名 A
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(QtWidgets.QLabel("源类别名 A:"))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setEditable(True)
        self.src_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.src_combo.setMinimumWidth(200)
        self.src_combo.setToolTip("目前数据集中已有的类别名，如 0-apple")
        src_row.addWidget(self.src_combo)
        src_row.addStretch()
        layout.addLayout(src_row)

        # 新类别名 B
        dst_row = QtWidgets.QHBoxLayout()
        dst_row.addWidget(QtWidgets.QLabel("新类别名 B:"))
        self.dst_edit = QtWidgets.QLineEdit()
        self.dst_edit.setPlaceholderText("要改成的新名称，如 apple")
        dst_row.addWidget(self.dst_edit, 1)
        layout.addLayout(dst_row)

        # 历史配置恢复
        self.history_bar = HistoryBar("label_rename", self._apply_config)
        layout.addWidget(self.history_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始重命名")
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
        current = self.src_combo.currentText()
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        self.src_combo.addItems(names)
        self.src_combo.setEditText(current if current else "")
        self.src_combo.blockSignals(False)

    def _restore_config(self):
        self._apply_config(load_config("label_rename"))

    def _apply_config(self, cfg):
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.src_combo.setEditText(str(cfg.get("src", "")))
        self.dst_edit.setText(cfg.get("dst", ""))
        labels_dir = cfg.get("labels_dir", "")
        if labels_dir:
            self._load_class_names(labels_dir)

    def _persist_config(self):
        save_config("label_rename", {
            "labels_dir": self.labels_edit.text().strip(),
            "src": self.src_combo.currentText().strip(),
            "dst": self.dst_edit.text().strip(),
        })

    def _start(self):
        labels_dir = self.labels_edit.text().strip()
        if not labels_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 labels 目录。")
            return
        src = self.src_combo.currentText().strip()
        dst = self.dst_edit.text().strip()
        if not src or not dst:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写源类别名 A 和新类别名 B。")
            return

        self._persist_config()
        self.log_view.clear()
        self.start_btn.setEnabled(False)

        self._worker = LabelRenameWorker(
            labels_dir=labels_dir,
            src_name=src,
            dst_name=dst,
        )
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)
        write_log(self.MODULE_TITLE, text)

    def _on_finished(self, ok):
        self.start_btn.setEnabled(True)
        if ok:
            QtWidgets.QMessageBox.information(self, "完成", "类别重命名完成。")
