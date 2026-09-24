"""背景添加模块：把背景图作为负样本加入数据集，降低误检。

做法（YOLO 负样本）：
1. 根据「背景占比」计算需要添加的背景图数量；
2. 背景图不足时通过随机数据增强（翻转/旋转/亮度/对比度/噪声）扩充；
3. 保留背景图原始分辨率（训练时再统一缩放）；
4. 复制到数据集 images，命名 bg_001、bg_002…；
5. 生成同名「空（0 字节）」标签文件到 labels。

关键：必须生成空的同名 txt，否则 ultralytics 会跳过该图并警告。
"""

import random
from pathlib import Path

import cv2
import numpy as np

from config import load as load_config, save as save_config
from log import write_log
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


def random_augment(img_bgr):
    """对背景图随机应用数据增强（翻转/旋转/亮度/对比度/噪声），返回新图。"""
    img = img_bgr.copy()
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.2:
        img = cv2.flip(img, 0)
    rot = random.choice([
        None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ])
    if rot is not None:
        img = cv2.rotate(img, rot)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = np.clip(
            (img.astype(np.float32) - 127.5) * factor + 127.5, 0, 255
        ).astype(np.uint8)
    if random.random() < 0.3:
        sigma = random.uniform(3, 12)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


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

    def __init__(self, bg_dir, dataset_dir, ratio, parent=None):
        super().__init__(parent)
        self.bg_dir = bg_dir
        self.dataset_dir = dataset_dir
        self.ratio = ratio              # 背景占比（百分比，1~49）

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

        images_dir = structure["train_images"]
        labels_dir = structure["train_labels"]
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 统计现有原图数量（排除已有的 bg_* 背景图）
        existing = sorted(
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
            and not p.stem.startswith("bg_")
        )
        n = len(existing)
        if n == 0:
            self.log.emit("数据集中没有找到图片。")
            self.finished.emit(0, 0)
            return

        # 计算需要的背景图数量：B / (N + B) = ratio / 100
        ratio = max(0, min(self.ratio, 49))
        need = int(round(n * ratio / (100 - ratio)))
        self.log.emit(f"原图 {n} 张，背景占比 {ratio}%，需要背景图 {need} 张。")

        bg_files = sorted(
            p for p in bg_dir.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        if not bg_files:
            self.log.emit("背景图文件夹中没有找到图片。")
            self.finished.emit(0, 0)
            return

        random.shuffle(bg_files)
        if len(bg_files) < need:
            self.log.emit(
                f"背景图只有 {len(bg_files)} 张，将通过随机数据增强扩充到 {need} 张。"
            )

        idx = self._next_bg_index(images_dir)
        ok = 0
        fail = 0
        for i in range(need):
            self.progress.emit(i + 1, need)
            src = bg_files[i % len(bg_files)]
            img = imread_unicode(src)
            if img is None:
                fail += 1
                self.log.emit(f"[失败] 无法读取背景图：{src}")
                continue
            if i >= len(bg_files):
                img = random_augment(img)  # 扩充部分做随机增强

            dst = images_dir / f"bg_{idx:03d}{src.suffix.lower() or '.jpg'}"
            if not imwrite_unicode(dst, img):
                fail += 1
                self.log.emit(f"[失败] 保存失败：{dst}")
                continue
            (labels_dir / f"bg_{idx:03d}.txt").touch()
            self.log.emit(
                f"[背景] {src.name}{'（增强）' if i >= len(bg_files) else ''} "
                f"-> {dst.name}（空标签已生成）"
            )
            ok += 1
            idx += 1

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
            "把背景图作为负样本加入数据集（降低误检）：按「背景占比」自动计算需添加的数量，\n"
            "背景图不足时用随机数据增强（翻转/旋转/亮度/对比度/噪声）扩充，保留原始分辨率、\n"
            "复制到 images，并生成同名「空（0 字节）」标签文件到 labels。建议占比 5~10%。"
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
        param_row.addWidget(QtWidgets.QLabel("背景占比 %:"))
        self.ratio_spin = QtWidgets.QSpinBox()
        self.ratio_spin.setRange(1, 49)
        self.ratio_spin.setValue(10)
        self.ratio_spin.setToolTip("背景图占整个数据集（原图+背景图）的百分比，建议 5~10%")
        param_row.addWidget(self.ratio_spin)
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
        self.ratio_spin.setValue(int(cfg.get("ratio", 10)))

    def _persist_config(self):
        save_config("background", {
            "bg_dir": self.bg_edit.text().strip(),
            "dataset_dir": self.dataset_edit.text().strip(),
            "ratio": self.ratio_spin.value(),
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

        self._persist_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = BackgroundWorker(
            bg_dir=bg_dir,
            dataset_dir=dataset_dir,
            ratio=self.ratio_spin.value(),
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

    def _on_finished(self, ok, fail):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"背景图添加完成：成功 {ok} 张，失败 {fail} 张。"
        )
