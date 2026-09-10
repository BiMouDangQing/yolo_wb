"""数据增强模块：对图片（及对应 YOLO 标签）做常见的数据增强。

支持两大类增强：

几何增强（会同步变换 YOLO 标签坐标）：
- 水平翻转 / 垂直翻转；
- 旋转 90°（顺时针）/ 180° / 270°（逆时针）。

色彩与噪声增强（标签内容不变，仅随图复制）：
- 亮度调整 / 对比度调整 / 饱和度调整 / 色调偏移；
- 高斯噪声 / 椒盐噪声。

处理逻辑复用 converter / white_balance 的模式：
QThread 后台处理，信号回传日志与预览；每个勾选的增强方式
对每张原图生成一张增强图（命名 `<原名>_<标签>.<后缀>`）。
提供 labels 目录时，会同步生成同名 `.txt` 标签，使增强结果
可直接用于 YOLO 训练。
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

# 几何增强：key -> 显示名（这些会同步变换标签坐标）
GEO_AUGS = [
    ("hflip", "水平翻转"),
    ("vflip", "垂直翻转"),
    ("rot90", "旋转 90°（顺时针）"),
    ("rot180", "旋转 180°"),
    ("rot270", "旋转 270°（逆时针）"),
]

# 色彩 / 噪声增强：key -> (显示名, 参数名, 最小值, 最大值, 默认值, 步长)
COLOR_AUGS = [
    ("brightness", "亮度调整", "亮度因子", 0.2, 2.0, 0.8, 0.05),
    ("contrast", "对比度调整", "对比度因子", 0.2, 2.0, 0.8, 0.05),
    ("saturation", "饱和度调整", "饱和度因子", 0.0, 2.0, 1.2, 0.05),
    ("hue", "色调偏移", "色调偏移", -90.0, 90.0, 10.0, 1.0),
    ("gaussian", "高斯噪声", "噪声强度", 1.0, 100.0, 15.0, 1.0),
    ("saltpepper", "椒盐噪声", "噪点比例", 0.001, 0.1, 0.01, 0.001),
]

# 需要同步变换标签坐标的增强 key 集合
GEO_KEYS = {key for key, _ in GEO_AUGS}


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径，cv2.imread 不支持）。"""
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


# ---------- 几何增强 ----------
def augment_hflip(img):
    return cv2.flip(img, 1)


def augment_vflip(img):
    return cv2.flip(img, 0)


def augment_rot90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def augment_rot180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def augment_rot270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ---------- 色彩 / 噪声增强 ----------
def augment_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_contrast(img, factor):
    out = (img.astype(np.float32) - 127.5) * factor + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_saturation(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_hue(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + delta) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_salt_pepper(img, prob):
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < prob / 2] = 255
    out[mask > 1 - prob / 2] = 0
    return out


# key -> 应用函数（几何函数忽略 param，色彩函数接收 param）
APPLY = {
    "hflip": lambda img, _p: augment_hflip(img),
    "vflip": lambda img, _p: augment_vflip(img),
    "rot90": lambda img, _p: augment_rot90(img),
    "rot180": lambda img, _p: augment_rot180(img),
    "rot270": lambda img, _p: augment_rot270(img),
    "brightness": augment_brightness,
    "contrast": augment_contrast,
    "saturation": augment_saturation,
    "hue": augment_hue,
    "gaussian": augment_gaussian_noise,
    "saltpepper": augment_salt_pepper,
}


def transform_yolo_line(line, key):
    """对 YOLO 标签一行做几何变换，返回新行文本。

    行格式：`class cx cy w h`（归一化 0~1）。
    水平翻转 / 垂直翻转 / 旋转 90 / 180 / 270 的坐标映射如下，
    其余 key 原样返回。
    """
    parts = line.split()
    if len(parts) < 5:
        return line
    cls = parts[0]
    try:
        cx, cy, w, h = (float(x) for x in parts[1:5])
    except ValueError:
        return line

    if key == "hflip":
        cx = 1.0 - cx
    elif key == "vflip":
        cy = 1.0 - cy
    elif key == "rot90":      # 顺时针 90°
        cx, cy, w, h = 1.0 - cy, cx, h, w
    elif key == "rot180":
        cx, cy = 1.0 - cx, 1.0 - cy
    elif key == "rot270":     # 逆时针 90°
        cx, cy, w, h = cy, 1.0 - cx, h, w
    else:
        return line

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class AugmentWorker(QtCore.QThread):
    """后台执行数据增强，避免阻塞界面。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)        # (成功数, 失败数)

    def __init__(self, images_dir, labels_dir, output_dir, ops, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.ops = ops  # [{"key": str, "label": str, "param": float|None}, ...]

    def run(self):
        images = Path(self.images_dir).expanduser()
        if not images.exists():
            self.log.emit(f"输入路径不存在：{images}")
            self.finished.emit(0, 0)
            return

        input_is_file = images.is_file()
        if input_is_file:
            if images.suffix.upper() not in SUPPORTED_EXTS:
                self.log.emit(f"输入文件格式不支持：{images}")
                self.finished.emit(0, 0)
                return
            files = [images]
        else:
            files = sorted(
                p for p in images.rglob("*")
                if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
            )

        if not files:
            self.log.emit("没有找到可处理的图片。")
            self.finished.emit(0, 0)
            return

        labels_dir = Path(self.labels_dir).expanduser() if self.labels_dir else None
        out = Path(self.output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        self.log.emit(
            f"共 {len(files)} 张图片，启用 {len(self.ops)} 种增强："
            + "、".join(op["label"] for op in self.ops)
        )

        ok = fail = 0
        thumbs = []
        thumb_labels = []
        for src in files:
            img = imread_unicode(src)
            if img is None:
                fail += 1
                self.log.emit(f"[失败] 无法读取：{src}")
                continue

            rel = src.name if input_is_file else src.relative_to(images)

            # 原图对应的标签内容（用于同步生成增强标签）
            if labels_dir is not None:
                label_src = labels_dir / rel.with_suffix(".txt")
            else:
                label_src = src.with_suffix(".txt")
            label_lines = None
            if label_src.is_file():
                try:
                    label_lines = label_src.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                except OSError:
                    label_lines = None

            for op in self.ops:
                key = op["key"]
                tag = op["key"]
                try:
                    result = APPLY[key](img, op.get("param"))
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    self.log.emit(f"[失败] 增强 {op['label']} 处理失败：{src}（{exc}）")
                    continue

                dst = out / rel.parent / (rel.stem + "_" + tag + rel.suffix)
                dst.parent.mkdir(parents=True, exist_ok=True)

                # 生成增强后的标签
                if label_lines is not None:
                    if key in GEO_KEYS:
                        new_lines = [
                            transform_yolo_line(line, key) for line in label_lines
                        ]
                    else:
                        new_lines = label_lines[:]
                    lbl_dst = dst.with_suffix(".txt")
                    try:
                        lbl_dst.write_text(
                            "\n".join(new_lines) + ("\n" if new_lines else ""),
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        self.log.emit(f"[警告] 标签写入失败：{lbl_dst}（{exc}）")

                if imwrite_unicode(dst, result):
                    ok += 1
                    self.log.emit(f"[成功] {src.name} --{op['label']}--> {dst.name}")
                else:
                    fail += 1
                    self.log.emit(f"[失败] 保存失败：{src}（{op['label']}）")
                    continue

                if len(thumbs) < 200:
                    try:
                        thumbs.append((make_thumb_bgr(img), make_thumb_bgr(result)))
                        thumb_labels.append(f"{src.name} · {op['label']}")
                    except Exception as exc:  # noqa: BLE001
                        self.log.emit(f"缩略图生成失败：{src}（{exc}）")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：成功 {ok} 张，失败 {fail} 张。")
        self.finished.emit(ok, fail)


class AugmentModule(QtWidgets.QWidget):
    MODULE_TITLE = "数据增强"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._checks = {}   # key -> QCheckBox
        self._params = {}   # key -> QDoubleSpinBox
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "对图片批量做数据增强，每个勾选的增强方式对每张原图生成一张新图。\n"
            "几何增强（翻转 / 旋转）会同步变换 YOLO 标签；色彩与噪声增强标签不变。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # images 目录
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(QtWidgets.QLabel("images 目录:"))
        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText("选择图片文件夹或单个图片文件")
        img_row.addWidget(self.images_edit, 1)
        img_btn = QtWidgets.QPushButton("浏览")
        img_btn.clicked.connect(self._pick_images)
        img_row.addWidget(img_btn)
        layout.addLayout(img_row)

        # labels 目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("labels 目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("可选，YOLO 标签文件夹（.txt 与图片同名）")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 输出目录
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("输出目录:"))
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("增强结果保存到该目录")
        out_row.addWidget(self.output_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # 几何增强分组
        geo_box = QtWidgets.QGroupBox("几何增强（会同步变换标签）")
        geo_layout = QtWidgets.QGridLayout(geo_box)
        for i, (key, label) in enumerate(GEO_AUGS):
            check = QtWidgets.QCheckBox(label)
            self._checks[key] = check
            geo_layout.addWidget(check, i // 2, i % 2)
        layout.addWidget(geo_box)

        # 色彩与噪声增强分组（带参数）
        color_box = QtWidgets.QGroupBox("色彩与噪声增强（标签不变）")
        color_layout = QtWidgets.QGridLayout(color_box)
        for i, (key, label, pname, lo, hi, default, step) in enumerate(COLOR_AUGS):
            check = QtWidgets.QCheckBox(label)
            self._checks[key] = check
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(3 if step < 0.01 else 2)
            spin.setValue(default)
            spin.setMinimumWidth(90)
            self._params[key] = spin
            color_layout.addWidget(check, i, 0)
            color_layout.addWidget(QtWidgets.QLabel(pname), i, 1)
            color_layout.addWidget(spin, i, 2)
        color_layout.setColumnStretch(3, 1)
        layout.addWidget(color_box)

        # 翻页预览（原图 / 增强结果）
        self.browser = PreviewBrowser(dual=True)
        layout.addWidget(self.browser, 1)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始增强")
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
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.images_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def _start(self):
        images_dir = self.images_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 images 目录或图片文件。")
            return
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输出目录。")
            return

        # 收集选中的增强方式（含参数）
        ops = []
        for key, label in GEO_AUGS:
            if self._checks[key].isChecked():
                ops.append({"key": key, "label": label, "param": None})
        for key, label, *_ in COLOR_AUGS:
            if self._checks[key].isChecked():
                ops.append({
                    "key": key,
                    "label": label,
                    "param": self._params[key].value(),
                })

        if not ops:
            QtWidgets.QMessageBox.warning(self, "提示", "请至少勾选一种增强方式。")
            return

        self.log_view.clear()
        self.log_view.appendPlainText("开始增强...")
        self.browser.clear()
        self.start_btn.setEnabled(False)

        self._worker = AugmentWorker(
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            output_dir=output_dir,
            ops=ops,
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
            self, "完成", f"增强完成：成功 {ok} 张，失败 {fail} 张。"
        )
