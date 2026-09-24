"""图片重命名模块：把图片按序号从指定起点开始连续命名。

命名格式：
    {前缀}{序号:0{位数}d}{原扩展名}
    例如前缀 img_、起始 1、位数 4 → img_0001.jpg、img_0002.jpg…

支持两种输出方式：
- 复制到新文件夹：原图不动，新命名文件写入输出目录（默认，更安全）；
- 原地重命名：直接改原图文件名（可同步重命名对应 YOLO 标签）。

同步标签：可选，自动检测 images 同级 labels 目录，把同名 .txt 同步重命名 / 复制。
"""

import re
import shutil
from pathlib import Path

from config import load as load_config, save as save_config
from qt_binding import QtCore, QtGui, QtWidgets, Signal

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def _natural_key(name):
    """自然排序键：把数字段转成整数，避免 2 > 10 这种字典序问题。"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def _find_labels_dir(images_dir):
    """自动检测与 images 对应的 labels 目录，找不到返回 None。"""
    p = Path(images_dir).expanduser()
    # 情况 1：images/train -> labels/train
    if p.name != "images":
        candidate = p.parent.parent / "labels" / p.name
        if candidate.is_dir():
            return candidate
    # 情况 2：images -> labels
    candidate = p.parent / "labels"
    if candidate.is_dir():
        return candidate
    return None


class RenameWorker(QtCore.QThread):
    """后台执行图片重命名，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    finished = Signal(int, int)      # (成功数, 失败数)

    def __init__(self, images_dir, labels_dir, output_dir, inplace,
                 start, width, prefix, sort_by, sync_labels, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.labels_dir = labels_dir          # 为空时自动检测
        self.output_dir = output_dir
        self.inplace = inplace                # True=原地重命名，False=复制到新文件夹
        self.start = start
        self.width = width
        self.prefix = prefix
        self.sort_by = sort_by                # name / mtime
        self.sync_labels = sync_labels

    def _new_name(self, seq):
        return f"{self.prefix}{seq:0{self.width}d}"

    def _sorted_files(self, images):
        files = [p for p in images.rglob("*")
                 if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS]
        if self.sort_by == "mtime":
            files.sort(key=lambda p: p.stat().st_mtime)
        else:
            files.sort(key=lambda p: _natural_key(p.name))
        return files

    def run(self):
        images = Path(self.images_dir).expanduser()
        if not images.is_dir():
            self.log.emit(f"图片文件夹不存在：{images}")
            self.finished.emit(0, 0)
            return

        files = self._sorted_files(images)
        if not files:
            self.log.emit("图片文件夹中没有找到图片。")
            self.finished.emit(0, 0)
            return

        if not self.inplace:
            out = Path(self.output_dir).expanduser()
            if not str(self.output_dir).strip():
                self.log.emit("复制模式下必须指定输出目录。")
                self.finished.emit(0, 0)
                return
            out.mkdir(parents=True, exist_ok=True)

        # 标签目录
        labels = None
        if self.sync_labels:
            if str(self.labels_dir).strip():
                labels = Path(self.labels_dir).expanduser()
            else:
                labels = _find_labels_dir(images)
            if labels is not None and not labels.is_dir():
                self.log.emit(f"标签目录不存在（将不同步标签）：{labels}")
                labels = None
            if labels is not None:
                self.log.emit(f"同步标签目录：{labels}")

        self.log.emit(f"共 {len(files)} 张图片，起始序号 {self.start}，"
                      f"补零 {self.width} 位，前缀 {self.prefix or '（无）'}，"
                      f"方式：{'原地重命名' if self.inplace else '复制到新文件夹'}。")

        # 复制模式下：目标目录与图片目录相同时，等同于原地重命名会互相覆盖，直接拒绝
        if not self.inplace and out.resolve() == images.resolve():
            self.log.emit("输出目录不能与图片文件夹相同（请换一个目录，或改用「原地重命名」）。")
            self.finished.emit(0, 0)
            return

        ok = 0
        fail = 0
        total = len(files)
        for i, src in enumerate(files):
            self.progress.emit(i + 1, total)
            seq = self.start + i
            new_stem = self._new_name(seq)
            ext = src.suffix.lower() or ".jpg"

            if self.inplace:
                dst_img = src.with_name(new_stem + ext)
            else:
                dst_img = out / (new_stem + ext)

            # 目标已存在且不是源文件本身 → 跳过，避免覆盖
            if dst_img.exists() and dst_img.resolve() != src.resolve():
                fail += 1
                self.log.emit(f"[跳过] 目标已存在：{dst_img.name}（来自 {src.name}）")
                continue

            try:
                if self.inplace:
                    shutil.move(str(src), str(dst_img))
                else:
                    shutil.copy2(src, dst_img)
                # 同步标签
                if labels is not None:
                    src_txt = labels / (src.stem + ".txt")
                    if src_txt.is_file():
                        if self.inplace:
                            dst_txt = labels / (new_stem + ".txt")
                            shutil.move(str(src_txt), str(dst_txt))
                        else:
                            out_labels = out.parent / "labels" \
                                if out.name == "images" else out / "labels"
                            out_labels.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_txt, out_labels / (new_stem + ".txt"))
                self.log.emit(f"[{i + 1}/{total}] {src.name} -> {dst_img.name}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                self.log.emit(f"[失败] {src.name}：{exc}")

        self.log.emit(f"\n完成：成功重命名 {ok} 张，失败/跳过 {fail} 张。")
        self.finished.emit(ok, fail)


class RenameModule(QtWidgets.QWidget):
    MODULE_TITLE = "图片重命名"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把图片按序号从指定起点开始连续命名，例如前缀 img_、起始 1、位数 4 → img_0001.jpg。\n"
            "可同步重命名同名 YOLO 标签（.txt），保证图片与标签配对不丢失。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 图片文件夹
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(QtWidgets.QLabel("图片文件夹:"))
        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText("图片文件夹（必填）")
        img_row.addWidget(self.images_edit, 1)
        img_btn = QtWidgets.QPushButton("浏览")
        img_btn.clicked.connect(self._pick_images)
        img_row.addWidget(img_btn)
        layout.addLayout(img_row)

        # 输出方式
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("输出方式:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["复制到新文件夹", "原地重命名"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # 输出目录（仅复制模式）
        self.out_row = QtWidgets.QHBoxLayout()
        self.out_row.addWidget(QtWidgets.QLabel("输出目录:"))
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("新命名图片的输出目录（复制模式必填）")
        self.out_row.addWidget(self.out_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_out)
        self.out_row.addWidget(out_btn)
        layout.addLayout(self.out_row)

        # 标签目录（可选）
        lab_row = QtWidgets.QHBoxLayout()
        lab_row.addWidget(QtWidgets.QLabel("标签目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("留空则自动检测 images 同级 labels 目录")
        lab_row.addWidget(self.labels_edit, 1)
        lab_btn = QtWidgets.QPushButton("浏览")
        lab_btn.clicked.connect(self._pick_labels)
        lab_row.addWidget(lab_btn)
        layout.addLayout(lab_row)

        # 参数
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("起始序号:"))
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, 1000000)
        self.start_spin.setValue(1)
        param_row.addWidget(self.start_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("补零位数:"))
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 8)
        self.width_spin.setValue(4)
        param_row.addWidget(self.width_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("前缀:"))
        self.prefix_edit = QtWidgets.QLineEdit()
        self.prefix_edit.setPlaceholderText("可选，如 img_")
        self.prefix_edit.setMaximumWidth(120)
        param_row.addWidget(self.prefix_edit)
        param_row.addStretch()
        layout.addLayout(param_row)

        # 排序 + 同步标签
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("排序方式:"))
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItem("修改时间", "mtime")
        self.sort_combo.addItem("文件名（自然排序）", "name")
        opt_row.addWidget(self.sort_combo)
        opt_row.addSpacing(16)
        self.sync_check = QtWidgets.QCheckBox("同步重命名同名标签（.txt）")
        self.sync_check.setChecked(True)
        opt_row.addWidget(self.sync_check)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

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

        self._on_mode_changed()

    def _pick_images(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.images_edit.setText(path)

    def _pick_out(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.out_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择标签目录")
        if path:
            self.labels_edit.setText(path)

    def _on_mode_changed(self):
        inplace = self.mode_combo.currentIndex() == 1
        self.out_row.itemAt(0).widget().setEnabled(not inplace)
        self.out_edit.setEnabled(not inplace)
        for i in range(1, self.out_row.count()):
            w = self.out_row.itemAt(i).widget()
            if w is not None:
                w.setEnabled(not inplace)

    def _restore_config(self):
        cfg = load_config("rename")
        self.images_edit.setText(cfg.get("images_dir", ""))
        self.out_edit.setText(cfg.get("output_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.mode_combo.setCurrentIndex(1 if cfg.get("inplace") else 0)
        self.start_spin.setValue(int(cfg.get("start", 1)))
        self.width_spin.setValue(int(cfg.get("width", 4)))
        self.prefix_edit.setText(cfg.get("prefix", ""))
        idx = self.sort_combo.findData(cfg.get("sort_by", "mtime"))
        self.sort_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.sync_check.setChecked(bool(cfg.get("sync_labels", True)))

    def _persist_config(self):
        save_config("rename", {
            "images_dir": self.images_edit.text().strip(),
            "output_dir": self.out_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "inplace": self.mode_combo.currentIndex() == 1,
            "start": self.start_spin.value(),
            "width": self.width_spin.value(),
            "prefix": self.prefix_edit.text().strip(),
            "sort_by": self.sort_combo.currentData(),
            "sync_labels": self.sync_check.isChecked(),
        })

    def _start(self):
        images_dir = self.images_edit.text().strip()
        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择图片文件夹。")
            return

        inplace = self.mode_combo.currentIndex() == 1
        output_dir = self.out_edit.text().strip()
        if not inplace and not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "复制模式下请选择输出目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = RenameWorker(
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            output_dir=output_dir,
            inplace=inplace,
            start=self.start_spin.value(),
            width=self.width_spin.value(),
            prefix=self.prefix_edit.text().strip(),
            sort_by=self.sort_combo.currentData(),
            sync_labels=self.sync_check.isChecked(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_finished(self, ok, fail):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.start_btn.setEnabled(True)
