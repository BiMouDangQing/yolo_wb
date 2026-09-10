"""废图筛选模块：检测模糊、过暗、过曝等低质量图片。

判据：
- 模糊：拉普拉斯方差低于阈值（值越小越模糊）；
- 过暗/全黑：平均亮度低于阈值；
- 过曝：亮度 > 250 的像素比例高于阈值。

处理方式：仅报告 / 移动到 _bad 文件夹 / 删除。
"""

from pathlib import Path

import cv2
import numpy as np

from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_bgr

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def analyze(img_bgr):
    """返回 (模糊度, 平均亮度, 过曝比例)。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    over = float((gray > 250).mean())
    return blur, mean, over


def judge(blur, mean, over, blur_th, dark_th, over_th):
    """返回 (是否废图, 原因)。"""
    if blur < blur_th:
        return True, f"模糊（拉普拉斯方差 {blur:.0f} < {blur_th}）"
    if mean < dark_th:
        return True, f"过暗（平均亮度 {mean:.1f} < {dark_th}）"
    if over > over_th:
        return True, f"过曝（亮像素比例 {over:.2f} > {over_th:.2f}）"
    return False, ""


class QualityWorker(QtCore.QThread):
    """后台执行废图筛选，避免阻塞界面。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 说明列表)
    finished = Signal(int)  # 发现的废图数

    def __init__(self, input_path, blur_th, dark_th, over_th, action, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.blur_th = blur_th
        self.dark_th = dark_th
        self.over_th = over_th
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

        self.log.emit(f"共找到 {len(files)} 张图片，正在分析...")

        bad_dir = src / "_bad"
        bad_count = 0
        thumbs = []
        thumb_labels = []
        for p in files:
            img = imread_unicode(p)
            if img is None:
                self.log.emit(f"[跳过] 无法读取 {p}")
                continue
            blur, mean, over = analyze(img)
            is_bad, reason = judge(
                blur, mean, over,
                self.blur_th, self.dark_th, self.over_th,
            )
            if not is_bad:
                continue

            bad_count += 1
            try:
                thumbs.append(make_thumb_bgr(img))
                thumb_labels.append(f"{p.name}：{reason}")
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"缩略图生成失败：{p}（{exc}）")
            if self.action == "report":
                self.log.emit(f"[废图] {p}：{reason}")
            elif self.action == "delete":
                try:
                    p.unlink()
                    self.log.emit(f"[已删除] {p}：{reason}")
                except OSError as exc:
                    self.log.emit(f"[删除失败] {p}：{exc}")
            elif self.action == "move":
                dst = bad_dir / p.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    p.rename(dst)
                    self.log.emit(f"[已移动] {p} -> {dst}：{reason}")
                except OSError as exc:
                    self.log.emit(f"[移动失败] {p}：{exc}")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：共 {len(files)} 张，发现废图 {bad_count} 张。")
        self.finished.emit(bad_count)


class QualityModule(QtWidgets.QWidget):
    MODULE_TITLE = "废图筛选"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "检测模糊、过暗、过曝的低质量图片。\n"
            "模糊阈值越大越严格；过暗阈值与过曝比例同理。"
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

        # 阈值
        th_row = QtWidgets.QHBoxLayout()
        th_row.addWidget(QtWidgets.QLabel("模糊阈值:"))
        self.blur_spin = QtWidgets.QSpinBox()
        self.blur_spin.setRange(0, 2000)
        self.blur_spin.setValue(100)
        th_row.addWidget(self.blur_spin)
        th_row.addSpacing(12)
        th_row.addWidget(QtWidgets.QLabel("过暗阈值:"))
        self.dark_spin = QtWidgets.QSpinBox()
        self.dark_spin.setRange(0, 255)
        self.dark_spin.setValue(15)
        th_row.addWidget(self.dark_spin)
        th_row.addSpacing(12)
        th_row.addWidget(QtWidgets.QLabel("过曝比例:"))
        self.over_spin = QtWidgets.QDoubleSpinBox()
        self.over_spin.setRange(0.0, 1.0)
        self.over_spin.setSingleStep(0.05)
        self.over_spin.setValue(0.6)
        th_row.addWidget(self.over_spin)
        th_row.addStretch()
        layout.addLayout(th_row)

        # 处理方式
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(QtWidgets.QLabel("处理方式:"))
        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.addItem("仅报告（不删除）", "report")
        self.action_combo.addItem("移动到 _bad 文件夹", "move")
        self.action_combo.addItem("删除废图", "delete")
        action_row.addWidget(self.action_combo)
        action_row.addStretch()
        layout.addLayout(action_row)

        # 翻页预览（废图 + 原因）
        self.browser = PreviewBrowser(dual=False)
        layout.addWidget(self.browser)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始筛选")
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
                "删除操作不可恢复！\n确定要删除检测出的废图吗？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        self.log_view.clear()
        self.log_view.appendPlainText("开始筛选...")
        self.browser.clear()
        self.start_btn.setEnabled(False)

        self._worker = QualityWorker(
            input_path=input_path,
            blur_th=self.blur_spin.value(),
            dark_th=self.dark_spin.value(),
            over_th=self.over_spin.value(),
            action=action,
        )
        self._worker.log.connect(self._append_log)
        self._worker.previews.connect(self._on_previews)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_previews(self, items, labels):
        self.browser.set_data(items, labels)

    def _on_finished(self, bad_count):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "完成", f"筛选完成，发现废图 {bad_count} 张。")
