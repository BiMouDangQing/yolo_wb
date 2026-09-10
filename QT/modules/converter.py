"""图片转换模块：把 HEIC / 其他格式图片批量转为 JPG。

复用 tools/jpg.py（通用图片 → JPG）与 tools/heic2jpg.py（HEIC/HEIF → JPG）。
"""

import importlib.util
from pathlib import Path

from PIL import Image, ImageOps

from config import load as load_config, save as save_config
from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_rgb

# 项目根/tools：QT/modules/converter.py -> parents[2] 即项目根目录
TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tools_module(name):
    """按文件名加载 tools 目录下的模块，避免与第三方同名模块冲突。"""
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ConverterWorker(QtCore.QThread):
    """后台执行图片转换，避免阻塞界面。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)  # (成功数, 失败数)

    def __init__(self, mode, input_path, output_dir, quality, replace_original, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.input_path = input_path
        self.output_dir = output_dir
        self.quality = quality
        self.replace_original = replace_original

    def run(self):
        src_path = Path(self.input_path).expanduser()

        if self.mode == "heic":
            try:
                mod = _load_tools_module("heic2jpg")
            except SystemExit as exc:
                self.log.emit(f"无法启用 HEIC 转换：{exc}")
                self.finished.emit(0, 0)
                return
            exts = mod.HEIC_EXTS
            convert = mod.convert_to_jpg
        else:
            mod = _load_tools_module("jpg")
            exts = mod.SUPPORTED_EXTS
            convert = mod.convert_image

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
            self.log.emit("没有找到可转换的图片。")
            self.finished.emit(0, 0)
            return

        if self.replace_original:
            output_root = base_dir
        else:
            output_root = Path(self.output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        ok = fail = 0
        thumbs = []
        thumb_labels = []
        for src in files:
            rel = src.name if input_is_file else src.relative_to(base_dir)
            dst = output_root / Path(rel).with_suffix(".jpg")

            if dst.resolve() == src.resolve():
                self.log.emit(f"[跳过] {src} 已是 JPG")
                continue

            if convert(src, dst, self.quality):
                ok += 1
                try:
                    with Image.open(src) as im:
                        orig_thumb = make_thumb_rgb(ImageOps.exif_transpose(im))
                    with Image.open(dst) as im:
                        result_thumb = make_thumb_rgb(im)
                    thumbs.append((orig_thumb, result_thumb))
                    thumb_labels.append(src.name)
                except Exception as exc:  # noqa: BLE001
                    self.log.emit(f"缩略图生成失败：{src}（{exc}）")
                if self.replace_original:
                    try:
                        src.unlink()
                        self.log.emit(f"[成功] {src} -> {dst}（原文件已删除）")
                    except OSError as exc:
                        self.log.emit(f"[成功] {src} -> {dst}（删除原文件失败：{exc}）")
                else:
                    self.log.emit(f"[成功] {src} -> {dst}")
            else:
                fail += 1
                self.log.emit(f"[失败] {src}")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：成功 {ok} 张，失败 {fail} 张。")
        self.finished.emit(ok, fail)


class ConverterModule(QtWidgets.QWidget):
    MODULE_TITLE = "图片转换"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 转换模式
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("转换模式:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("通用图片 → JPG", "jpg")
        self.mode_combo.addItem("HEIC/HEIF → JPG", "heic")
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # 输入路径
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(QtWidgets.QLabel("输入路径:"))
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("选择或输入图片文件夹 / 单个图片文件")
        input_layout.addWidget(self.input_edit)
        folder_btn = QtWidgets.QPushButton("文件夹")
        folder_btn.clicked.connect(self._pick_input_dir)
        input_layout.addWidget(folder_btn)
        file_btn = QtWidgets.QPushButton("文件")
        file_btn.clicked.connect(self._pick_input_file)
        input_layout.addWidget(file_btn)
        layout.addLayout(input_layout)

        # 保存路径
        output_layout = QtWidgets.QHBoxLayout()
        output_layout.addWidget(QtWidgets.QLabel("保存路径:"))
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("保存到该文件夹")
        output_layout.addWidget(self.output_edit)
        self.output_btn = QtWidgets.QPushButton("浏览")
        self.output_btn.clicked.connect(self._pick_output_dir)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)

        # 选项
        option_layout = QtWidgets.QHBoxLayout()
        self.replace_check = QtWidgets.QCheckBox("替换原文件（原地转换并删除原图）")
        self.replace_check.toggled.connect(self._on_replace_toggled)
        option_layout.addWidget(self.replace_check)
        option_layout.addStretch()
        option_layout.addWidget(QtWidgets.QLabel("质量:"))
        self.quality_spin = QtWidgets.QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setSuffix(" %")
        option_layout.addWidget(self.quality_spin)
        layout.addLayout(option_layout)

        # 翻页预览（原图 / 结果）
        self.browser = PreviewBrowser(dual=True)
        layout.addWidget(self.browser)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始转换")
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

    def _pick_input_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_edit.setText(path)

    def _pick_input_file(self):
        if self.mode_combo.currentData() == "heic":
            filter_str = "图片文件 (*.heic *.heif *.hif *.avif);;所有文件 (*)"
        else:
            filter_str = (
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff "
                "*.gif *.ppm *.pgm *.pbm *.ico *.jpe *.jfif);;所有文件 (*)"
            )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片文件", "", filter_str)
        if path:
            self.input_edit.setText(path)

    def _pick_output_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if path:
            self.output_edit.setText(path)

    def _on_replace_toggled(self, checked):
        self.output_edit.setEnabled(not checked)
        self.output_btn.setEnabled(not checked)

    def _restore_config(self):
        """恢复上次保存的路径与参数。"""
        cfg = load_config("converter")
        self.input_edit.setText(cfg.get("input_path", ""))
        self.output_edit.setText(cfg.get("output_dir", ""))
        self.quality_spin.setValue(int(cfg.get("quality", 95)))
        self.replace_check.setChecked(bool(cfg.get("replace", False)))
        mode = cfg.get("mode", "jpg")
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def _persist_config(self):
        """保存当前路径与参数。"""
        save_config("converter", {
            "mode": self.mode_combo.currentData(),
            "input_path": self.input_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "quality": self.quality_spin.value(),
            "replace": self.replace_check.isChecked(),
        })

    def _start(self):
        input_path = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        replace = self.replace_check.isChecked()

        if not input_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写输入路径。")
            return
        if not replace and not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写保存路径，或勾选“替换原文件”。")
            return

        self._persist_config()
        self.log_view.clear()
        self.log_view.appendPlainText("开始转换...")
        self.browser.clear()
        self.start_btn.setEnabled(False)

        self._worker = ConverterWorker(
            mode=self.mode_combo.currentData(),
            input_path=input_path,
            output_dir=output_dir,
            quality=self.quality_spin.value(),
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
        QtWidgets.QMessageBox.information(self, "完成", f"转换完成：成功 {ok} 张，失败 {fail} 张。")
