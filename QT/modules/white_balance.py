"""图像白平衡模块：对单张图片或整个文件夹做自动/手动白平衡校正。

支持的算法：
- 灰度世界（Gray World）：假设图像各通道均值相等，按均值缩放 RGB；
- 白斑法（White Patch / Max RGB）：以各通道最大值为参考进行缩放；
- 完美反射（Perfect Reflector）：以各通道 95% 分位数为参考，抗高光更稳；
- 手动增益（Manual）：用户手动调整 R / G / B 通道增益。

处理逻辑复用 converter 的模式：QThread 后台处理，信号回传日志与预览。
"""

from pathlib import Path

import cv2
import numpy as np

from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_bgr

# 可处理的图片格式（cv2 可解码的常见格式）
SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

# 算法标识 -> 显示名
METHODS = [
    ("gray_world", "灰度世界（Gray World）"),
    ("max_white", "白斑法（White Patch）"),
    ("perfect_reflector", "完美反射（Perfect Reflector）"),
    ("manual", "手动增益（Manual）"),
]


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径，cv2.imread 不支持）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片（支持中文等非 ASCII 路径）。

    先写入同目录临时文件，再原子替换到目标路径，
    避免覆盖过程中出错导致原文件损坏或未替换。
    """
    path = Path(path)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    tmp = path.with_name(path.name + ".tmp")
    try:
        buf.tofile(str(tmp))
        tmp.replace(path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _gain_factors(img_bgr, method, manual_gains):
    """计算各通道缩放因子（B, G, R 顺序）。"""
    b, g, r = cv2.split(img_bgr.astype(np.float64))

    if method == "manual":
        return tuple(manual_gains)

    if method == "gray_world":
        mb, mg, mr = b.mean(), g.mean(), r.mean()
        ref = (mb + mg + mr) / 3.0
    elif method == "max_white":
        mb, mg, mr = b.max(), g.max(), r.max()
        ref = max(mb, mg, mr)
    else:  # perfect_reflector
        mb = np.percentile(b, 95)
        mg = np.percentile(g, 95)
        mr = np.percentile(r, 95)
        ref = max(mb, mg, mr)

    return (
        ref / max(mb, 1e-6),
        ref / max(mg, 1e-6),
        ref / max(mr, 1e-6),
    )


def apply_white_balance(img_bgr, method="gray_world", manual_gains=(1.0, 1.0, 1.0)):
    """对 BGR 图像做白平衡，返回处理后的 BGR 图像。"""
    factors = _gain_factors(img_bgr, method, manual_gains)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    b *= factors[0]
    g *= factors[1]
    r *= factors[2]
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


class WhiteBalanceWorker(QtCore.QThread):
    """后台执行白平衡处理，避免阻塞界面。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)       # (成功数, 失败数)

    def __init__(self, input_path, method, gains, output_dir,
                 replace_original, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.method = method
        self.gains = gains
        self.output_dir = output_dir
        self.replace_original = replace_original

    def run(self):
        src_path = Path(self.input_path).expanduser()

        input_is_file = src_path.is_file()
        if input_is_file:
            if src_path.suffix.upper() not in SUPPORTED_EXTS:
                self.log.emit(f"输入文件格式不支持：{src_path}")
                self.finished.emit(0, 0)
                return
            files = [src_path]
            base_dir = src_path.parent
        else:
            if not src_path.is_dir():
                self.log.emit(f"输入路径不存在：{src_path}")
                self.finished.emit(0, 0)
                return
            files = sorted(
                p for p in src_path.rglob("*")
                if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
            )
            base_dir = src_path

        if not files:
            self.log.emit("没有找到可处理的图片。")
            self.finished.emit(0, 0)
            return

        # 确定输出目录；为空时仅预览、不保存
        if self.replace_original:
            output_root = base_dir
        elif self.output_dir:
            output_root = Path(self.output_dir).expanduser()
            output_root.mkdir(parents=True, exist_ok=True)
        else:
            output_root = None

        ok = fail = 0
        thumbs = []
        thumb_labels = []
        for src in files:
            img = imread_unicode(src)
            if img is None:
                fail += 1
                self.log.emit(f"[失败] 无法读取：{src}")
                continue

            result = apply_white_balance(img, self.method, self.gains)

            try:
                thumbs.append((make_thumb_bgr(img), make_thumb_bgr(result)))
                thumb_labels.append(src.name)
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"缩略图生成失败：{src}（{exc}）")

            if output_root is None:
                ok += 1
                self.log.emit(f"[处理] {src}（仅预览，未保存）")
                continue

            rel = src.name if input_is_file else src.relative_to(base_dir)
            dst = output_root / rel
            # 未勾选“替换原文件”且输出目录就是原目录时，避免覆盖原图
            if dst.resolve() == src.resolve() and not self.replace_original:
                dst = output_root / (rel.stem + "_wb" + rel.suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)

            if imwrite_unicode(dst, result):
                ok += 1
                if self.replace_original:
                    self.log.emit(f"[成功] 已原地替换：{src}")
                else:
                    self.log.emit(f"[成功] {src} -> {dst}")
            else:
                fail += 1
                self.log.emit(f"[失败] 保存失败：{src}")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：成功 {ok} 张，失败 {fail} 张。")
        self.finished.emit(ok, fail)


class WhiteBalanceModule(QtWidgets.QWidget):
    MODULE_TITLE = "白平衡"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 算法选择
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("算法:"))
        self.method_combo = QtWidgets.QComboBox()
        for key, label in METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self.method_combo)
        method_row.addStretch()
        layout.addLayout(method_row)

        # 手动增益（仅“手动增益”算法启用）
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(QtWidgets.QLabel("手动增益:"))
        self.gain_spins = {}
        for key, label in (("R", "R"), ("G", "G"), ("B", "B")):
            gain_row.addWidget(QtWidgets.QLabel(label))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.1, 3.0)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            spin.setDecimals(2)
            self.gain_spins[key] = spin
            gain_row.addWidget(spin)
        gain_row.addStretch()
        layout.addLayout(gain_row)

        # 输入路径
        input_row = QtWidgets.QHBoxLayout()
        input_row.addWidget(QtWidgets.QLabel("输入路径:"))
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("选择或输入图片文件夹 / 单个图片文件")
        input_row.addWidget(self.input_edit, 1)
        folder_btn = QtWidgets.QPushButton("文件夹")
        folder_btn.clicked.connect(self._pick_input_dir)
        input_row.addWidget(folder_btn)
        file_btn = QtWidgets.QPushButton("文件")
        file_btn.clicked.connect(self._pick_input_file)
        input_row.addWidget(file_btn)
        layout.addLayout(input_row)

        # 保存选项
        save_row = QtWidgets.QHBoxLayout()
        self.replace_check = QtWidgets.QCheckBox("替换原文件（原地覆盖）")
        self.replace_check.toggled.connect(self._on_replace_toggled)
        save_row.addWidget(self.replace_check)
        save_row.addWidget(QtWidgets.QLabel("保存路径:"))
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("留空则仅预览不保存")
        save_row.addWidget(self.output_edit, 1)
        self.output_btn = QtWidgets.QPushButton("浏览")
        self.output_btn.clicked.connect(self._pick_output_dir)
        save_row.addWidget(self.output_btn)
        layout.addLayout(save_row)

        # 翻页预览（原图 / 结果）
        self.browser = PreviewBrowser(dual=True)
        layout.addWidget(self.browser, 1)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始处理")
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

    # ---------- 事件 ----------
    def _on_method_changed(self, _index):
        manual = self.method_combo.currentData() == "manual"
        for spin in self.gain_spins.values():
            spin.setEnabled(manual)

    def _on_replace_toggled(self, checked):
        self.output_edit.setEnabled(not checked)
        self.output_btn.setEnabled(not checked)

    def _pick_input_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_edit.setText(path)

    def _pick_input_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;所有文件 (*)"
        )
        if path:
            self.input_edit.setText(path)

    def _pick_output_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if path:
            self.output_edit.setText(path)

    def _start(self):
        input_path = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        replace = self.replace_check.isChecked()

        if not input_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写输入路径。")
            return
        # 文件夹模式必须有保存方式，单张图片可仅预览
        src_path = Path(input_path).expanduser()
        if not replace and not output_dir and not src_path.is_file():
            QtWidgets.QMessageBox.warning(
                self, "提示",
                "文件夹模式请填写保存路径，或勾选“替换原文件”。\n"
                "单张图片可留空仅预览。"
            )
            return

        self.log_view.clear()
        self.log_view.appendPlainText("开始处理...")
        self.browser.clear()
        self.start_btn.setEnabled(False)

        # 手动增益按 BGR 顺序传给 worker
        gains = (
            self.gain_spins["B"].value(),
            self.gain_spins["G"].value(),
            self.gain_spins["R"].value(),
        )

        self._worker = WhiteBalanceWorker(
            input_path=input_path,
            method=self.method_combo.currentData(),
            gains=gains,
            output_dir=output_dir,
            replace_original=replace,
        )
        self._worker.log.connect(self._append_log)
        self._worker.previews.connect(self._on_previews)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_previews(self, items, labels):
        self.browser.set_data(items, labels)

    def _on_finished(self, ok, fail):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"处理完成：成功 {ok} 张，失败 {fail} 张。"
        )
