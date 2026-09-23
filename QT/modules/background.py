"""背景添加模块：把背景图作为负样本加入数据集，降低误检。

做法（YOLO 负样本）：
1. 从背景图文件夹随机抽取 N 张；
2. 缩放到目标分辨率（默认 1280，与现有数据集一致）；
3. 复制到数据集的 images/train（或 images）下，命名 bg_001、bg_002…；
4. 生成同名「空（0 字节）」标签文件到 labels/train（或 labels）下。

关键：必须生成空的同名 txt，否则 ultralytics 会跳过该图并警告。
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import load as load_config, save as save_config
from qt_binding import QtCore, QtGui, QtWidgets, Signal

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


def resize_to_target(img_bgr, target):
    """把最长边缩放到 target，保持纵横比。"""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= 0 or longest == target:
        return img_bgr
    scale = target / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def detect_structure(dataset_dir):
    """检测数据集结构，返回 dict 或 None。

    - 已划分：dataset/images/train 等（YOLO 标准结构）；
    - 未划分：dataset/images 平铺。
    """
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        return None
    if (d / "images" / "train").is_dir():
        return {
            "split": True,
            "train_images": d / "images" / "train",
            "train_labels": d / "labels" / "train",
            "val_images": d / "images" / "val" if (d / "images" / "val").is_dir() else None,
            "val_labels": d / "labels" / "val" if (d / "labels" / "val").is_dir() else None,
        }
    if (d / "images").is_dir():
        return {
            "split": False,
            "train_images": d / "images",
            "train_labels": d / "labels",
            "val_images": None,
            "val_labels": None,
        }
    return None


class BackgroundWorker(QtCore.QThread):
    """后台执行背景图添加，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    finished = Signal(int, int)      # (成功数, 失败数)

    def __init__(self, bg_dir, dataset_dir, train_count, val_count,
                 target_size, parent=None):
        super().__init__(parent)
        self.bg_dir = bg_dir
        self.dataset_dir = dataset_dir
        self.train_count = train_count
        self.val_count = val_count
        self.target_size = target_size

    def _next_bg_index(self, images_dir):
        """扫描现有 bg_* 文件，返回下一个可用序号。"""
        idx = 1
        for p in images_dir.glob("bg_*"):
            try:
                num = int(p.stem[3:])
                idx = max(idx, num + 1)
            except ValueError:
                continue
        return idx

    def run(self):
        bg_dir = Path(self.bg_dir).expanduser()
        if not bg_dir.is_dir():
            self.log.emit(f"背景图文件夹不存在：{bg_dir}")
            self.finished.emit(0, 0)
            return

        structure = detect_structure(self.dataset_dir)
        if structure is None:
            self.log.emit(f"数据集目录不存在或结构无法识别：{self.dataset_dir}")
            self.finished.emit(0, 0)
            return

        bg_files = sorted(
            p for p in bg_dir.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        if not bg_files:
            self.log.emit("背景图文件夹中没有找到图片。")
            self.finished.emit(0, 0)
            return

        need = self.train_count + self.val_count
        if len(bg_files) < need:
            self.log.emit(
                f"背景图只有 {len(bg_files)} 张，少于请求的 {need} 张，将全部使用。"
            )
            need = len(bg_files)

        random.shuffle(bg_files)
        train_pool = bg_files[:self.train_count]
        val_pool = bg_files[self.train_count:self.train_count + self.val_count]

        self.log.emit(
            f"数据集结构：{'已划分（train/val）' if structure['split'] else '未划分（平铺）'}；"
            f"目标分辨率 {self.target_size}；"
            f"训练集添加 {len(train_pool)} 张，验证集添加 {len(val_pool)} 张。"
        )

        total = len(train_pool) + len(val_pool)
        done = 0
        ok = 0
        fail = 0

        # 训练集
        for bg in train_pool:
            done += 1
            self.progress.emit(done, total)
            img = imread_unicode(bg)
            if img is None:
                fail += 1
                self.log.emit(f"[失败] 无法读取背景图：{bg}")
                continue
            resized = resize_to_target(img, self.target_size)
            idx = self._next_bg_index(structure["train_images"])
            structure["train_images"].mkdir(parents=True, exist_ok=True)
            structure["train_labels"].mkdir(parents=True, exist_ok=True)
            dst = structure["train_images"] / f"bg_{idx:03d}{bg.suffix.lower() or '.jpg'}"
            if not imwrite_unicode(dst, resized):
                fail += 1
                self.log.emit(f"[失败] 保存失败：{dst}")
                continue
            (structure["train_labels"] / f"bg_{idx:03d}.txt").touch()
            self.log.emit(f"[train] {bg.name} -> {dst.name}（空标签已生成）")
            ok += 1

        # 验证集
        if val_pool and structure["val_images"] and structure["val_labels"]:
            structure["val_images"].mkdir(parents=True, exist_ok=True)
            structure["val_labels"].mkdir(parents=True, exist_ok=True)
            for bg in val_pool:
                done += 1
                self.progress.emit(done, total)
                img = imread_unicode(bg)
                if img is None:
                    fail += 1
                    self.log.emit(f"[失败] 无法读取背景图：{bg}")
                    continue
                resized = resize_to_target(img, self.target_size)
                idx = self._next_bg_index(structure["val_images"])
                dst = structure["val_images"] / f"bg_{idx:03d}{bg.suffix.lower() or '.jpg'}"
                if not imwrite_unicode(dst, resized):
                    fail += 1
                    self.log.emit(f"[失败] 保存失败：{dst}")
                    continue
                (structure["val_labels"] / f"bg_{idx:03d}.txt").touch()
                self.log.emit(f"[val] {bg.name} -> {dst.name}（空标签已生成）")
                ok += 1
        elif val_pool:
            self.log.emit("提示：数据集未划分或没有 val 目录，验证集背景图未添加。")

        self.log.emit(f"\n完成：成功添加 {ok} 张背景图，失败 {fail} 张。")
        self.finished.emit(ok, fail)


class BackgroundModule(QtWidgets.QWidget):
    MODULE_TITLE = "背景添加"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把背景图作为负样本加入数据集（降低误检）：随机抽取、缩放到目标分辨率、\n"
            "复制到 images，并生成同名「空（0 字节）」标签文件到 labels。\n"
            "建议背景图占训练集 5~10%，验证集可少量添加。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 背景图文件夹
        bg_row = QtWidgets.QHBoxLayout()
        bg_row.addWidget(QtWidgets.QLabel("背景图文件夹:"))
        self.bg_edit = QtWidgets.QLineEdit()
        self.bg_edit.setPlaceholderText("背景图片文件夹（必填）")
        bg_row.addWidget(self.bg_edit, 1)
        bg_btn = QtWidgets.QPushButton("浏览")
        bg_btn.clicked.connect(self._pick_bg)
        bg_row.addWidget(bg_btn)
        layout.addLayout(bg_row)

        # 数据集目录
        ds_row = QtWidgets.QHBoxLayout()
        ds_row.addWidget(QtWidgets.QLabel("数据集目录:"))
        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setPlaceholderText("数据集根目录（含 images/train 或 images 平铺）")
        ds_row.addWidget(self.dataset_edit, 1)
        ds_btn = QtWidgets.QPushButton("浏览")
        ds_btn.clicked.connect(self._pick_dataset)
        ds_row.addWidget(ds_btn)
        layout.addLayout(ds_row)

        # 参数
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("训练集数量:"))
        self.train_spin = QtWidgets.QSpinBox()
        self.train_spin.setRange(0, 100000)
        self.train_spin.setValue(60)
        param_row.addWidget(self.train_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("验证集数量:"))
        self.val_spin = QtWidgets.QSpinBox()
        self.val_spin.setRange(0, 100000)
        self.val_spin.setValue(0)
        param_row.addWidget(self.val_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("目标分辨率:"))
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(64, 8192)
        self.size_spin.setValue(1280)
        param_row.addWidget(self.size_spin)
        param_row.addStretch()
        layout.addLayout(param_row)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始添加")
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

    def _pick_bg(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择背景图文件夹")
        if path:
            self.bg_edit.setText(path)

    def _pick_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.dataset_edit.setText(path)

    def _restore_config(self):
        cfg = load_config("background")
        self.bg_edit.setText(cfg.get("bg_dir", ""))
        self.dataset_edit.setText(cfg.get("dataset_dir", ""))
        self.train_spin.setValue(int(cfg.get("train_count", 60)))
        self.val_spin.setValue(int(cfg.get("val_count", 0)))
        self.size_spin.setValue(int(cfg.get("target_size", 1280)))

    def _persist_config(self):
        save_config("background", {
            "bg_dir": self.bg_edit.text().strip(),
            "dataset_dir": self.dataset_edit.text().strip(),
            "train_count": self.train_spin.value(),
            "val_count": self.val_spin.value(),
            "target_size": self.size_spin.value(),
        })

    def _start(self):
        bg_dir = self.bg_edit.text().strip()
        dataset_dir = self.dataset_edit.text().strip()
        if not bg_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择背景图文件夹。")
            return
        if not dataset_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择数据集目录。")
            return
        if self.train_spin.value() == 0 and self.val_spin.value() == 0:
            QtWidgets.QMessageBox.warning(self, "提示", "训练集或验证集数量至少填一个。")
            return

        self._persist_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = BackgroundWorker(
            bg_dir=bg_dir,
            dataset_dir=dataset_dir,
            train_count=self.train_spin.value(),
            val_count=self.val_spin.value(),
            target_size=self.size_spin.value(),
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
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"背景图添加完成：成功 {ok} 张，失败 {fail} 张。"
        )
