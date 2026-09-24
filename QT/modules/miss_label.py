"""漏标检测模块：用 YOLO 模型找「模型框出来了、但标签 txt 里没有」的疑似漏标。

做法（标注清理第 2 层）：
1. 用指定模型（.pt）对图片批量推理（conf 可调）；
2. 读取每张图的 YOLO 标签，把标签框与模型框按 IoU 匹配；
3. 模型框与任何标签框都不重叠（IoU < 阈值）的，视为「疑似漏标」；
4. 输出疑似漏标明细 CSV，并翻页预览带模型框的图片，供人工确认补框。

依赖 ultralytics（模型推理）。
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from config import load as load_config, save as save_config
from log import write_log
from modules._history import HistoryBar
from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._annotation import AnnotationCanvas
from modules._preview import PreviewBrowser, make_thumb_bgr

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_label_boxes(path):
    """读取 YOLO 标签，返回 [(cls, x1, y1, x2, y2)]（归一化坐标）。"""
    boxes = []
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
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    except OSError:
        pass
    return boxes


def iou(a, b):
    """两个 (x1, y1, x2, y2) 框的 IoU。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(img_bgr, boxes, color=(0, 0, 255)):
    """在图上画出疑似漏标框（像素坐标）。"""
    out = img_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out


class MissLabelWorker(QtCore.QThread):
    """后台执行漏标检测（模型推理 + 标签对比）。"""

    log = Signal(str)
    progress = Signal(int, int)          # (当前进度, 总数)
    previews = Signal(object, object)    # (缩略图列表, 说明列表)
    stats = Signal(object)               # 统计结果 dict
    finished = Signal(int, int)          # (图片数, 疑似漏标框数)

    def __init__(self, model_path, images_dir, labels_dir, conf, iou_th,
                 csv_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.conf = conf
        self.iou_th = iou_th
        self.csv_path = csv_path
        self.miss_items = []  # 疑似漏标图片信息，供标注使用

    def run(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"无法加载 ultralytics：{exc}")
            self.finished.emit(0, 0)
            return

        self.log.emit(f"正在加载模型：{self.model_path}")
        try:
            model = YOLO(self.model_path)
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"模型加载失败：{exc}")
            self.finished.emit(0, 0)
            return

        images = Path(self.images_dir).expanduser()
        if not images.is_dir():
            self.log.emit(f"images 目录不存在：{images}")
            self.finished.emit(0, 0)
            return

        files = sorted(
            p for p in images.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        if not files:
            self.log.emit("images 目录中没有找到图片。")
            self.finished.emit(0, 0)
            return

        labels_dir = Path(self.labels_dir).expanduser() if self.labels_dir else None
        names = getattr(model, "names", {})
        self.log.emit(
            f"共 {len(files)} 张图片，conf={self.conf}，IoU 阈值={self.iou_th}。"
        )

        miss_count = 0
        miss_images = 0
        thumbs = []
        thumb_labels = []
        rows = []

        for i, img_path in enumerate(files, 1):
            self.progress.emit(i, len(files))
            rel = img_path.relative_to(images)

            # 读取标签框
            if labels_dir is not None:
                label_path = labels_dir / rel.with_suffix(".txt")
            else:
                label_path = img_path.with_suffix(".txt")
            label_boxes = read_label_boxes(label_path) if label_path.is_file() else []

            # 模型推理
            try:
                result = model(str(img_path), conf=self.conf, verbose=False)[0]
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"[失败] 推理出错 {img_path.name}：{exc}")
                continue

            h, w = result.orig_shape[:2]
            misses = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    # 转归一化 xyxy
                    nb = (xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h)
                    matched = any(iou(nb, lb[1:]) >= self.iou_th for lb in label_boxes)
                    if not matched:
                        misses.append((cls, conf, xyxy, nb))

            if misses:
                miss_images += 1
                miss_count += len(misses)
                self.miss_items.append({
                    "img_path": str(img_path),
                    "label_path": str(label_path),
                    "label_boxes": label_boxes,
                    "names": names,
                })
                for cls, conf, xyxy, nb in misses:
                    name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                    self.log.emit(
                        f"[疑似漏标] {img_path.name}：{name} conf={conf:.2f} "
                        f"框=({xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f})"
                    )
                    rows.append([
                        img_path.name, name, f"{conf:.3f}",
                        f"{nb[0]:.4f}", f"{nb[1]:.4f}", f"{nb[2]:.4f}", f"{nb[3]:.4f}",
                    ])
                # 预览：画框
                try:
                    img = imread_unicode(img_path)
                    if img is not None:
                        draw = draw_boxes(img, [m[2] for m in misses])
                        thumbs.append(make_thumb_bgr(draw))
                        thumb_labels.append(f"{img_path.name}（{len(misses)} 个疑似漏标）")
                except Exception as exc:  # noqa: BLE001
                    self.log.emit(f"缩略图生成失败：{img_path.name}（{exc}）")

        # 生成 CSV
        csv_path = None
        if self.csv_path:
            csv_path = Path(self.csv_path).expanduser()
            if csv_path.suffix.lower() != ".csv":
                csv_path = csv_path / "miss_labels.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "图片", "类别", "置信度",
                    "归一化x1", "归一化y1", "归一化x2", "归一化y2",
                ])
                for row in rows:
                    writer.writerow(row)
            self.log.emit(f"已生成疑似漏标明细 CSV：{csv_path}")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.stats.emit({
            "total": len(files),
            "miss_images": miss_images,
            "miss_count": miss_count,
            "csv_path": str(csv_path) if csv_path else "",
        })
        self.log.emit(
            f"\n完成：共 {len(files)} 张图片，疑似漏标 {miss_count} 处，"
            f"涉及 {miss_images} 张图片。"
        )
        self.finished.emit(len(files), miss_count)


class MissLabelModule(QtWidgets.QWidget):
    MODULE_TITLE = "漏标检测"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._miss_items = []
        self._current_label_path = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "用模型对图片批量推理，把「模型框出来了、但标签 txt 里没有」的框\n"
            "挑出来作为疑似漏标，人工确认后补框，可有效提升 recall。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 模型文件
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("模型文件:"))
        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("选择 .pt 模型（如 best.pt）")
        model_row.addWidget(self.model_edit, 1)
        model_btn = QtWidgets.QPushButton("浏览")
        model_btn.clicked.connect(self._pick_model)
        model_row.addWidget(model_btn)
        layout.addLayout(model_row)

        # images 目录
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(QtWidgets.QLabel("images 目录:"))
        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText("待检测图片文件夹")
        img_row.addWidget(self.images_edit, 1)
        img_btn = QtWidgets.QPushButton("浏览")
        img_btn.clicked.connect(self._pick_images)
        img_row.addWidget(img_btn)
        layout.addLayout(img_row)

        # labels 目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("labels 目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("留空表示与图片同目录的 .txt")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 参数
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("置信度 conf:"))
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.3)
        param_row.addWidget(self.conf_spin)
        param_row.addSpacing(12)
        param_row.addWidget(QtWidgets.QLabel("IoU 阈值:"))
        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.3)
        param_row.addWidget(self.iou_spin)
        param_row.addStretch()
        layout.addLayout(param_row)

        # CSV 输出
        csv_row = QtWidgets.QHBoxLayout()
        csv_row.addWidget(QtWidgets.QLabel("CSV 输出:"))
        self.csv_edit = QtWidgets.QLineEdit()
        self.csv_edit.setPlaceholderText("留空表示不生成 CSV")
        csv_row.addWidget(self.csv_edit, 1)
        csv_btn = QtWidgets.QPushButton("浏览")
        csv_btn.clicked.connect(self._pick_csv)
        csv_row.addWidget(csv_btn)
        layout.addLayout(csv_row)

        # 预览 / 标注 切换
        self.stack = QtWidgets.QStackedWidget()
        self.browser = PreviewBrowser(dual=False)
        self.stack.addWidget(self.browser)

        annot_page = QtWidgets.QWidget()
        annot_layout = QtWidgets.QVBoxLayout(annot_page)
        annot_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.shapeAdded.connect(self._on_shape_added)
        annot_layout.addWidget(self.canvas, 1)
        annot_row = QtWidgets.QHBoxLayout()
        annot_row.addWidget(QtWidgets.QLabel("类别:"))
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        annot_row.addWidget(self.class_combo, 1)
        undo_btn = QtWidgets.QPushButton("撤销")
        undo_btn.clicked.connect(self.canvas.undo)
        annot_row.addWidget(undo_btn)
        self.save_btn = QtWidgets.QPushButton("保存标注")
        self.save_btn.clicked.connect(self._save_annotation)
        annot_row.addWidget(self.save_btn)
        back_btn = QtWidgets.QPushButton("返回预览")
        back_btn.clicked.connect(self._back_to_preview)
        annot_row.addWidget(back_btn)
        annot_layout.addLayout(annot_row)
        self.stack.addWidget(annot_page)

        layout.addWidget(self.stack, 1)

        # 标注当前图按钮
        self.annotate_btn = QtWidgets.QPushButton("标注当前图")
        self.annotate_btn.setEnabled(False)
        self.annotate_btn.clicked.connect(self._enter_annotate)
        layout.addWidget(self.annotate_btn)

        # 历史配置恢复
        self.history_bar = HistoryBar("miss_label", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始检测")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._start)
        layout.addWidget(self.start_btn)

        # 日志
        layout.addWidget(QtWidgets.QLabel("日志:"))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

    def _pick_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt);;所有文件 (*)"
        )
        if path:
            self.model_edit.setText(path)

    def _pick_images(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 images 目录")
        if path:
            self.images_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择 CSV 保存位置", "", "CSV 文件 (*.csv)"
        )
        if path:
            self.csv_edit.setText(path)

    def _restore_config(self):
        self._apply_config(load_config("miss_label"))

    def _apply_config(self, cfg):
        self.model_edit.setText(cfg.get("model_path", ""))
        self.images_edit.setText(cfg.get("images_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.csv_edit.setText(cfg.get("csv_path", ""))
        self.conf_spin.setValue(float(cfg.get("conf", 0.3)))
        self.iou_spin.setValue(float(cfg.get("iou_th", 0.3)))

    def _persist_config(self):
        save_config("miss_label", {
            "model_path": self.model_edit.text().strip(),
            "images_dir": self.images_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "csv_path": self.csv_edit.text().strip(),
            "conf": self.conf_spin.value(),
            "iou_th": self.iou_spin.value(),
        })

    def _start(self):
        model_path = self.model_edit.text().strip()
        images_dir = self.images_edit.text().strip()
        if not model_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择模型文件。")
            return
        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 images 目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.browser.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = MissLabelWorker(
            model_path=model_path,
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            conf=self.conf_spin.value(),
            iou_th=self.iou_spin.value(),
            csv_path=self.csv_edit.text().strip(),
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
        if self._worker is not None:
            self._miss_items = self._worker.miss_items

    def _on_finished(self, total, miss_count):
        self.start_btn.setEnabled(True)
        self.annotate_btn.setEnabled(miss_count > 0)
        QtWidgets.QMessageBox.information(
            self, "完成",
            f"检测完成：共 {total} 张图片，疑似漏标 {miss_count} 处。\n"
            "请翻页查看带框预览，点「标注当前图」直接在图上补框。",
        )

    # ---------- 标注模式 ----------
    def _enter_annotate(self):
        """进入标注模式，加载当前预览的图片到标注画布。"""
        if not self._miss_items:
            return
        idx = getattr(self.browser, "_index", 0)
        if idx < 0 or idx >= len(self._miss_items):
            idx = 0
        item = self._miss_items[idx]

        img = imread_unicode(item["img_path"])
        if img is None:
            QtWidgets.QMessageBox.warning(self, "提示", f"无法读取图片：{item['img_path']}")
            return

        # 填充类别下拉框
        names = item["names"]
        self.class_combo.clear()
        if isinstance(names, dict):
            for k in sorted(names):
                self.class_combo.addItem(str(names[k]), int(k))
        elif isinstance(names, (list, tuple)):
            for i, n in enumerate(names):
                self.class_combo.addItem(str(n), int(i))
        else:
            self.class_combo.addItem("0", 0)

        # 显示原图 + 已有标签框（绿色）
        names_dict = {}
        if isinstance(names, dict):
            names_dict = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            names_dict = {i: str(n) for i, n in enumerate(names)}
        self.canvas.set_image(img, [lb[1:] for lb in item["label_boxes"]], names_dict)
        self._current_label_path = item["label_path"]

        # 默认选中第一个类别
        if self.class_combo.count() > 0:
            self.class_combo.setCurrentIndex(0)
            cls = self.class_combo.currentData()
            if cls is not None:
                self.canvas.set_current_class(int(cls))

        self.stack.setCurrentIndex(1)
        self.log_view.appendPlainText(
            f"[标注模式] 当前图片：{Path(item['img_path']).name}，"
            f"在图上拖拽画框，右键或「撤销」可删除上一个框"
        )

    def _on_class_changed(self, _index):
        cls = self.class_combo.currentData()
        if cls is not None:
            self.canvas.set_current_class(int(cls))

    def _on_shape_added(self, cx, cy, w, h):
        """画布画完一个框：画布已记录，这里只打日志。"""
        cls = self.class_combo.currentData()
        if cls is None:
            cls = 0
        self.log_view.appendPlainText(
            f"[新标注] class={cls} 框=({cx:.4f},{cy:.4f},{w:.4f},{h:.4f})"
        )

    def _save_annotation(self):
        """把画布上的新框追加到对应 YOLO txt。"""
        if not self._current_label_path:
            return
        new_boxes = self.canvas.get_new_boxes()
        if not new_boxes:
            QtWidgets.QMessageBox.information(self, "提示", "还没有画新框。")
            return
        path = Path(self._current_label_path)
        lines = [
            f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            for cls, cx, cy, w, h in new_boxes
        ]
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "错误", f"保存失败：{exc}")
            return
        self.log_view.appendPlainText(
            f"[已保存] {path.name} 追加 {len(lines)} 个框"
        )
        QtWidgets.QMessageBox.information(
            self, "完成", f"已把 {len(lines)} 个框追加到：\n{path}"
        )

    def _back_to_preview(self):
        """返回预览视图。"""
        self.stack.setCurrentIndex(0)
