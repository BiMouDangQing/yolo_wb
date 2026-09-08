"""图片去重模块：基于感知哈希（dHash）找出重复或高度相似的图片。

- 对每张图计算 64 位差异哈希（dHash）；
- 两两比较汉明距离，小于等于阈值即视为重复；
- 按文件名排序，每组保留第一张，其余标记为重复；
- 处理方式：仅报告 / 移动到 _duplicates 文件夹 / 删除。
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from qt_binding import QtCore, QtGui, QtWidgets, Signal

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def dhash_bytes(path, hash_size=8):
    """计算图片的 dHash 指纹（8 字节），支持中文路径与 EXIF 方向。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    # 每行相邻像素比较，得到 hash_size*hash_size 个 bit
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    return np.packbits(bits)


def hamming(a, b):
    """两个 dHash 指纹之间的汉明距离。"""
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


class DedupWorker(QtCore.QThread):
    """后台执行图片去重，避免阻塞界面。"""

    log = Signal(str)
    finished = Signal(int)  # 发现的重复图片数

    def __init__(self, input_path, threshold, action, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.threshold = threshold
        self.action = action  # report / move / delete

    def run(self):
        src = Path(self.input_path).expanduser()
        if not src.is_dir():
            self.log.emit(f"输入文件夹不存在：{src}")
            self.finished.emit(0)
            return

        files = sorted(
            p for p in src.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        if not files:
            self.log.emit("没有找到可处理的图片。")
            self.finished.emit(0)
            return

        self.log.emit(f"共找到 {len(files)} 张图片，正在计算指纹...")

        hashes = []
        for p in files:
            try:
                hashes.append(dhash_bytes(p))
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"[跳过] 无法读取 {p}：{exc}")
                hashes.append(None)

        # 贪心分组：把每张图归入第一个相似的代表图组
        groups = []   # list[list[Path]]
        reps = []     # 每组代表图的指纹
        for p, h in zip(files, hashes):
            if h is None:
                continue
            for i, rep in enumerate(reps):
                if hamming(h, rep) <= self.threshold:
                    groups[i].append(p)
                    break
            else:
                groups.append([p])
                reps.append(h)

        dup_dir = src / "_duplicates"
        dup_count = 0
        for g in groups:
            if len(g) <= 1:
                continue
            keep, dups = g[0], g[1:]
            self.log.emit(f"[重复组] 保留：{keep.name}，重复 {len(dups)} 张")
            for d in dups:
                dup_count += 1
                if self.action == "report":
                    self.log.emit(f"    [重复] {d}")
                elif self.action == "delete":
                    try:
                        d.unlink()
                        self.log.emit(f"    [已删除] {d}")
                    except OSError as exc:
                        self.log.emit(f"    [删除失败] {d}：{exc}")
                elif self.action == "move":
                    dst = dup_dir / d.relative_to(src)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        d.rename(dst)
                        self.log.emit(f"    [已移动] {d} -> {dst}")
                    except OSError as exc:
                        self.log.emit(f"    [移动失败] {d}：{exc}")

        self.log.emit(f"\n完成：共 {len(files)} 张，发现重复 {dup_count} 张。")
        self.finished.emit(dup_count)


class DedupModule(QtWidgets.QWidget):
    MODULE_TITLE = "图片去重"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "扫描文件夹，用感知哈希找出重复/高度相似的图片，每组保留第一张。\n"
            "汉明距离阈值越小越严格（0=完全相同才算重复）。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 输入路径
        input_row = QtWidgets.QHBoxLayout()
        input_row.addWidget(QtWidgets.QLabel("输入文件夹:"))
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("选择要扫描的图片文件夹")
        input_row.addWidget(self.input_edit, 1)
        browse_btn = QtWidgets.QPushButton("浏览")
        browse_btn.clicked.connect(self._pick_input)
        input_row.addWidget(browse_btn)
        layout.addLayout(input_row)

        # 选项
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("汉明距离阈值:"))
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(0, 64)
        self.threshold_spin.setValue(5)
        opt_row.addWidget(self.threshold_spin)
        opt_row.addStretch()
        opt_row.addWidget(QtWidgets.QLabel("处理方式:"))
        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.addItem("仅报告（不删除）", "report")
        self.action_combo.addItem("移动到 _duplicates 文件夹", "move")
        self.action_combo.addItem("删除重复图片", "delete")
        opt_row.addWidget(self.action_combo)
        layout.addLayout(opt_row)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始扫描")
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

    def _pick_input(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_edit.setText(path)

    def _start(self):
        input_path = self.input_edit.text().strip()
        if not input_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输入文件夹。")
            return

        action = self.action_combo.currentData()
        if action == "delete":
            ret = QtWidgets.QMessageBox.question(
                self, "确认删除",
                "删除操作不可恢复！\n确定要删除重复图片吗？\n（每组会保留第一张）",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        self.log_view.clear()
        self.log_view.appendPlainText("开始扫描...")
        self.start_btn.setEnabled(False)

        self._worker = DedupWorker(
            input_path=input_path,
            threshold=self.threshold_spin.value(),
            action=action,
        )
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_finished(self, dup_count):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "完成", f"扫描完成，发现重复 {dup_count} 张。")
