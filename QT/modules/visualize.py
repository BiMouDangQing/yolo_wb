"""标签可视化模块：把 YOLO 标签（检测框 / 分割多边形）绘制到图片上。

自动识别标签格式（按每行字段数）：
- 5 字段：检测框  `class cx cy w h`
- >5 奇数：分割多边形 `class x1 y1 x2 y2 ...`

类别名：优先读 labels 同级（或上级）的 classes.txt，找不到则显示 class_id。
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import load as load_config, save as save_config
from logs import write_log
from modules._history import HistoryBar
from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_bgr

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}

PALETTE = [
    (255, 60, 60), (60, 200, 90), (60, 120, 255), (255, 180, 40),
    (180, 60, 255), (60, 220, 220), (255, 120, 200), (160, 200, 40),
    (240, 140, 60), (100, 160, 255), (200, 80, 80), (80, 200, 140),
]


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def find_font():
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def load_font(size):
    path = find_font()
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def find_classes_txt(labels_dir):
    d = Path(labels_dir).expanduser()
    for _ in range(3):
        p = d / "classes.txt"
        if p.is_file():
            return p
        if d.parent == d:
            break
        d = d.parent
    return None


def load_class_names(labels_dir):
    classes_txt = find_classes_txt(labels_dir)
    if classes_txt is None:
        return []
    try:
        return [
            line.strip()
            for line in classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    except OSError:
        return []


def parse_label_file(path):
    """解析 YOLO 标签，返回 [(cls, kind, values)]。"""
    items = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if len(vals) == 4:
                items.append((cls, "detect", vals))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                items.append((cls, "seg", vals))
    except OSError:
        pass
    return items


def draw_labels(img_bgr, items, class_names):
    """把标签绘制到 BGR 图上，返回新 BGR 图。"""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    base = max(h, w)
    line_w = max(2, int(round(base / 640)))
    font_size = max(14, int(round(base / 48)))
    font = load_font(font_size)

    for cls, kind, vals in items:
        color = PALETTE[cls % len(PALETTE)]
        if kind == "detect":
            cx, cy, bw, bh = vals
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            label_pos = (x1, y1 - font_size - 4)
        else:
            pts = []
            for i in range(0, len(vals), 2):
                pts.append((vals[i] * w, vals[i + 1] * h))
            if len(pts) >= 3:
                draw.polygon(pts, outline=color, width=line_w)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label_pos = (min(xs), min(ys) - font_size - 4)

        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        text = f"{cls} {name}"
        bbox = draw.textbbox(label_pos, text, font=font)
        draw.rectangle(bbox, fill=(*color, 220))
        draw.text(label_pos, text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def find_labels_dir(images_dir):
    p = Path(images_dir).expanduser()
    candidate = p.parent / "labels"
    return candidate if candidate.is_dir() else None


class VisualizeWorker(QtCore.QThread):
    """后台执行标签可视化，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)      # (成功数, 无标签数)

    def __init__(self, images_dir, labels_dir, output_dir, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir

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

        labels = Path(self.labels_dir).expanduser() if self.labels_dir else find_labels_dir(images)
        if labels is None or not labels.is_dir():
            self.log.emit(f"标签目录不存在：{labels}")
            self.finished.emit(0, 0)
            return

        out = Path(self.output_dir).expanduser()
        if not str(self.output_dir).strip():
            self.log.emit("请指定输出目录。")
            self.finished.emit(0, 0)
            return
        out.mkdir(parents=True, exist_ok=True)

        class_names = load_class_names(labels)
        if class_names:
            self.log.emit(f"类别名：{class_names}")

        ok = 0
        no_label = 0
        thumbs = []
        thumb_labels = []
        total = len(files)
        for i, src in enumerate(files):
            self.progress.emit(i + 1, total)
            rel = src.relative_to(images)

            img = imread_unicode(src)
            if img is None:
                self.log.emit(f"[失败] 无法读取 {rel}")
                continue

            lbl = labels / rel.with_suffix(".txt")
            items = parse_label_file(lbl) if lbl.is_file() else []
            if not items:
                no_label += 1
                self.log.emit(f"[无标签] {rel}")

            annotated = draw_labels(img, items, class_names)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite_unicode(dst, annotated):
                self.log.emit(f"[失败] 保存失败 {dst}")
                continue

            ok += 1
            self.log.emit(f"[{i + 1}/{total}] {rel}（{len(items)} 个标签）")

            if len(thumbs) < 200:
                thumbs.append((make_thumb_bgr(img), make_thumb_bgr(annotated)))
                thumb_labels.append(f"{rel}（{len(items)} 个标签）")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：可视化 {ok} 张" + (f"，无标签 {no_label} 张。" if no_label else "。"))
        self.finished.emit(ok, no_label)


class VisualizeModule(QtWidgets.QWidget):
    MODULE_TITLE = "标签可视化"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把 YOLO 标签（检测框 / 分割多边形）绘制到图片上，方便肉眼检查标注。\n"
            "自动识别标签格式，类别名优先读 classes.txt，支持中文显示。"
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
        self.output_edit.setPlaceholderText("可视化图片保存位置（必填）")
        out_row.addWidget(self.output_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # 翻页预览（原图 / 标注后）
        self.browser = PreviewBrowser(dual=True)
        layout.addWidget(self.browser)

        # 历史配置恢复
        self.history_bar = HistoryBar("visualize", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始可视化")
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
        self._apply_config(load_config("visualize"))

    def _apply_config(self, cfg):
        self.images_edit.setText(cfg.get("images_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.output_edit.setText(cfg.get("output_dir", ""))

    def _persist_config(self):
        save_config("visualize", {
            "images_dir": self.images_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
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

        self._worker = VisualizeWorker(
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            output_dir=output_dir,
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

    def _on_finished(self, ok, no_label):
        self.start_btn.setEnabled(True)
        msg = f"可视化完成：成功 {ok} 张。"
        if no_label:
            msg += f"\n其中 {no_label} 张无标签。"
        QtWidgets.QMessageBox.information(self, "完成", msg)
