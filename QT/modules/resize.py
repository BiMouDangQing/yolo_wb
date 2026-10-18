"""图片缩放模块：把图片缩放到指定尺寸（最长边），生成新数据集，保留 polygon 标签。

YOLO（检测 / 分割）标签都是归一化坐标（0~1），图片等比缩放后标签值不变，
因此标签直接原样复制，无需重新计算。

输出结构（保持相对路径镜像）：
    <输出目录>/images/<相对路径>      缩放后的图片
    <输出目录>/labels/<相对路径>      复制的标签（可选）
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from config import load as load_config, save as save_config
from logs import write_log
from modules._history import HistoryBar
from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_bgr

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """保存图片（支持中文等非 ASCII 路径）。"""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def resize_longest(img_bgr, target):
    """把最长边缩放到 target，保持纵横比；已更小或相等则不放大。"""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= 0:
        return img_bgr
    scale = target / longest
    if scale >= 1.0:
        return img_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def find_labels_dir(images_dir):
    """自动检测与 images 对应的 labels 目录，找不到返回 None。"""
    p = Path(images_dir).expanduser()
    candidate = p.parent / "labels"
    if candidate.is_dir():
        return candidate
    return None


class ResizeWorker(QtCore.QThread):
    """后台执行图片缩放，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)      # (成功数, 失败数)

    def __init__(self, images_dir, labels_dir, output_dir, target_size,
                 sync_labels, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.labels_dir = labels_dir          # 为空时自动检测
        self.output_dir = output_dir
        self.target_size = target_size
        self.sync_labels = sync_labels

    def run(self):
        images = Path(self.images_dir).expanduser()
        if not images.is_dir():
            self.log.emit(f"图片目录不存在：{images}")
            self.finished.emit(0, 0)
            return

        files = sorted(
            p for p in images.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        if not files:
            self.log.emit("图片目录中没有找到图片。")
            self.finished.emit(0, 0)
            return

        out = Path(self.output_dir).expanduser()
        if not str(self.output_dir).strip():
            self.log.emit("请指定输出目录。")
            self.finished.emit(0, 0)
            return
        if out.resolve() == images.resolve():
            self.log.emit("输出目录不能与图片目录相同。")
            self.finished.emit(0, 0)
            return
        out_images = out / "images"
        out_labels = out / "labels"

        # 标签目录
        labels = None
        if self.sync_labels:
            if str(self.labels_dir).strip():
                labels = Path(self.labels_dir).expanduser()
            else:
                labels = find_labels_dir(images)
            if labels is not None and not labels.is_dir():
                self.log.emit(f"标签目录不存在（将不同步标签）：{labels}")
                labels = None
            if labels is not None:
                self.log.emit(f"同步标签目录：{labels}")

        self.log.emit(f"共 {len(files)} 张图片，目标尺寸（最长边）{self.target_size}。")

        ok = 0
        fail = 0
        no_label = 0
        thumbs = []
        thumb_labels = []
        total = len(files)
        for i, src in enumerate(files):
            self.progress.emit(i + 1, total)
            rel = src.relative_to(images)

            img = imread_unicode(src)
            if img is None:
                fail += 1
                self.log.emit(f"[失败] 无法读取 {rel}")
                continue

            resized = resize_longest(img, self.target_size)
            dst = out_images / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite_unicode(dst, resized):
                fail += 1
                self.log.emit(f"[失败] 保存失败 {dst}")
                continue

            # 同步标签（归一化坐标不变，直接复制）
            if labels is not None:
                lbl_src = labels / rel.with_suffix(".txt")
                if lbl_src.is_file():
                    lbl_dst = out_labels / rel.with_suffix(".txt")
                    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(lbl_src, lbl_dst)
                else:
                    no_label += 1
                    self.log.emit(f"[无标签] {rel}")

            ok += 1
            self.log.emit(
                f"[{i + 1}/{total}] {rel}（{img.shape[1]}x{img.shape[0]} → "
                f"{resized.shape[1]}x{resized.shape[0]}）"
            )

            # 预览：原图 vs 缩放后（限制前 200 张）
            if len(thumbs) < 200:
                thumbs.append((make_thumb_bgr(img), make_thumb_bgr(resized)))
                thumb_labels.append(f"{rel}（{img.shape[1]}x{img.shape[0]} → {resized.shape[1]}x{resized.shape[0]}）")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(
            f"\n完成：缩放 {ok} 张，失败 {fail} 张" +
            (f"，缺标签 {no_label} 张。" if labels is not None else "。")
        )
        self.finished.emit(ok, fail)


class ResizeModule(QtWidgets.QWidget):
    MODULE_TITLE = "图片缩放"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把图片缩放到指定尺寸（最长边，默认 1280），生成新数据集，保留 polygon 标签。\n"
            "YOLO 标签为归一化坐标，等比缩放后标签值不变，因此标签直接原样复制。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 图片目录
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(QtWidgets.QLabel("图片目录:"))
        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText("图片文件夹（必填）")
        img_row.addWidget(self.images_edit, 1)
        img_btn = QtWidgets.QPushButton("浏览")
        img_btn.clicked.connect(self._pick_images)
        img_row.addWidget(img_btn)
        layout.addLayout(img_row)

        # 标签目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("标签目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("留空自动检测 images 同级 labels 目录")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 输出目录
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("输出目录:"))
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("缩放后新数据集保存位置（必填）")
        out_row.addWidget(self.output_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # 参数
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("目标尺寸（最长边）:"))
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(64, 8192)
        self.size_spin.setValue(1280)
        param_row.addWidget(self.size_spin)
        param_row.addSpacing(16)
        self.sync_check = QtWidgets.QCheckBox("同步复制标签（.txt）")
        self.sync_check.setChecked(True)
        param_row.addWidget(self.sync_check)
        param_row.addStretch()
        layout.addLayout(param_row)

        # 翻页预览（原图 / 缩放后）
        self.browser = PreviewBrowser(dual=True)
        layout.addWidget(self.browser)

        # 历史配置恢复
        self.history_bar = HistoryBar("resize", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始缩放")
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

    def _pick_images(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片目录")
        if path:
            self.images_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择标签目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def _restore_config(self):
        self._apply_config(load_config("resize"))

    def _apply_config(self, cfg):
        self.images_edit.setText(cfg.get("images_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.output_edit.setText(cfg.get("output_dir", ""))
        self.size_spin.setValue(int(cfg.get("target_size", 1280)))
        self.sync_check.setChecked(bool(cfg.get("sync_labels", True)))

    def _persist_config(self):
        save_config("resize", {
            "images_dir": self.images_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "target_size": self.size_spin.value(),
            "sync_labels": self.sync_check.isChecked(),
        })

    def _start(self):
        images_dir = self.images_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择图片目录。")
            return
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输出目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.browser.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = ResizeWorker(
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            output_dir=output_dir,
            target_size=self.size_spin.value(),
            sync_labels=self.sync_check.isChecked(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.previews.connect(self._on_previews)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)
        write_log(self.MODULE_TITLE, text)

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_previews(self, items, labels):
        self.browser.set_data(items, labels)

    def _on_finished(self, ok, fail):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"缩放完成：成功 {ok} 张，失败 {fail} 张。"
        )
