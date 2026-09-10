"""过曝处理模块：对过曝图片做高光压缩、降低亮度。

复用 tools/overexposure.py 中的算法，后台 QThread 处理，不阻塞界面。
"""

import importlib.util
from pathlib import Path

from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_bgr

# 项目根/tools：QT/modules/overexposure.py -> parents[2] 即项目根目录
TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tools_module(name):
    """按文件名加载 tools 目录下的模块，避免与第三方同名模块冲突。"""
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OverexposureWorker(QtCore.QThread):
    """后台执行过曝校正。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)       # (成功数, 失败数)

    def __init__(self, input_path, output_dir, strength,
                 replace_original, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.strength = strength
        self.replace_original = replace_original

    def run(self):
        mod = _load_tools_module("overexposure")
        correct = mod.correct_overexposure
        imread = mod.imread_unicode
        imwrite = mod.imwrite_unicode
        exts = mod.SUPPORTED_EXTS

        src_path = Path(self.input_path).expanduser()
        input_is_file = src_path.is_file()
        if input_is_file:
            if src_path.suffix.upper() not in exts:
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
                if p.is_file() and p.suffix.upper() in exts
            )
            base_dir = src_path

        if not files:
            self.log.emit("没有找到可处理的图片。")
            self.finished.emit(0, 0)
            return

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
            img = imread(src)
            if img is None:
                fail += 1
                self.log.emit(f"[失败] 无法读取：{src}")
                continue

            out = correct(img, self.strength)

            try:
                thumbs.append((make_thumb_bgr(img), make_thumb_bgr(out)))
                thumb_labels.append(src.name)
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"缩略图生成失败：{src}（{exc}）")

            if output_root is None:
                ok += 1
                self.log.emit(f"[处理] {src}（仅预览，未保存）")
                continue

            rel = src.name if input_is_file else src.relative_to(base_dir)
            dst = output_root / rel
            if dst.resolve() == src.resolve() and not self.replace_original:
                dst = output_root / (rel.stem + "_fixed" + rel.suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)

            if imwrite(dst, out):
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


class OverexposureModule(QtWidgets.QWidget):
    MODULE_TITLE = "过曝处理"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "对过曝图片做高光压缩，降低整体亮度。\n"
            "强度越大压暗越明显，建议 30~70。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 强度
        strength_row = QtWidgets.QHBoxLayout()
        strength_row.addWidget(QtWidgets.QLabel("校正强度:"))
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        self.strength_value = QtWidgets.QLabel("50")
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_value.setText(str(v))
        )
        strength_row.addWidget(self.strength_slider, 1)
        strength_row.addWidget(self.strength_value)
        layout.addLayout(strength_row)

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

        self._worker = OverexposureWorker(
            input_path=input_path,
            output_dir=output_dir,
            strength=self.strength_slider.value(),
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
