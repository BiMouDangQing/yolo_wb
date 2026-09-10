"""XML 标注转 YOLO 模块：把 Pascal VOC / LabelImg 的 XML 标注转成 YOLO txt，并筛选无标注图片。

复用 tools/xml2yolo.py 中的解析与转换函数，后台 QThread 执行，避免阻塞界面。
"""

import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import load as load_config, save as save_config
from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_bgr

# 项目根/tools：QT/modules/xml2yolo.py -> parents[2] 即项目根目录
TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tools_module(name):
    """按文件名加载 tools 目录下的模块，避免与第三方同名模块冲突。"""
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def draw_boxes(img_bgr, boxes):
    """在 BGR 图上画出标注框，用于验证转换。"""
    out = img_bgr.copy()
    for name, xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(out, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    return out


class Xml2YoloWorker(QtCore.QThread):
    """后台执行 XML 转 YOLO 与无标注筛选。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int, int)  # (转换数, 失败数, 无标注数)

    def __init__(self, images_dir, xml_dir, labels_dir, unlabeled_dir,
                 classes, move_unlabeled, no_unlabeled, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.xml_dir = xml_dir
        self.labels_dir = labels_dir
        self.unlabeled_dir = unlabeled_dir
        self.classes = classes
        self.move_unlabeled = move_unlabeled
        self.no_unlabeled = no_unlabeled

    def run(self):
        mod = _load_tools_module("xml2yolo")
        supported = mod.SUPPORTED_EXTS
        parse_xml = mod.parse_xml
        read_image_size = mod.read_image_size
        to_yolo_line = mod.to_yolo_line
        resolve_classes = mod.resolve_classes

        images_dir = Path(self.images_dir).expanduser()
        if not images_dir.is_dir():
            self.log.emit(f"图片目录不存在：{images_dir}")
            self.finished.emit(0, 0, 0)
            return

        xml_dir = Path(self.xml_dir).expanduser() if self.xml_dir else images_dir
        if not xml_dir.is_dir():
            self.log.emit(f"XML 目录不存在：{xml_dir}")
            self.finished.emit(0, 0, 0)
            return

        # 收集图片
        images = {
            p.stem: p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.upper() in supported
        }
        if not images:
            self.log.emit("图片目录中没有找到图片。")
            self.finished.emit(0, 0, 0)
            return

        # 收集并解析 XML
        xml_files = sorted(
            p for p in xml_dir.rglob("*")
            if p.is_file() and p.suffix.upper() == ".XML"
        )
        if not xml_files:
            self.log.emit("没有找到 XML 标注文件。")
            self.finished.emit(0, 0, 0)
            return

        records = []
        for xml_path in xml_files:
            try:
                records.append((xml_path, *parse_xml(xml_path)))
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"[跳过] XML 解析失败 {xml_path}：{exc}")

        class_map = resolve_classes(records, self.classes)

        labels_dir = Path(self.labels_dir).expanduser() if self.labels_dir else None
        if labels_dir:
            labels_dir.mkdir(parents=True, exist_ok=True)

        ok = fail = 0
        thumbs = []
        thumb_labels = []
        annotated_stems = set()
        for xml_path, filename, width, height, boxes in records:
            annotated_stems.add(xml_path.stem)
            if filename:
                annotated_stems.add(Path(filename).stem)

            # XML 缺少 size 时从图片回退读取尺寸
            if not width or not height:
                img_path = images.get(xml_path.stem)
                if img_path:
                    try:
                        width, height = read_image_size(img_path)
                    except Exception as exc:  # noqa: BLE001
                        self.log.emit(f"[失败] 无法读取图片尺寸 {img_path}：{exc}")

            if not width or not height:
                fail += 1
                self.log.emit(f"[失败] 缺少尺寸信息：{xml_path}")
                continue

            lines = []
            for name, xmin, ymin, xmax, ymax in boxes:
                line = to_yolo_line(name, xmin, ymin, xmax, ymax, width, height, class_map)
                if line is not None:
                    lines.append(line)
                else:
                    self.log.emit(f"[警告] 未知类别 '{name}'（{xml_path.name}），已忽略")

            dst = (labels_dir / (xml_path.stem + ".txt")) if labels_dir else xml_path.with_suffix(".txt")
            dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            ok += 1
            self.log.emit(f"[成功] {xml_path.name} -> {dst.name}（{len(lines)} 个框）")

            img_path = images.get(xml_path.stem)
            if img_path:
                try:
                    img = imread_unicode(img_path)
                    if img is not None:
                        annotated = draw_boxes(img, boxes)
                        thumbs.append(make_thumb_bgr(annotated))
                        thumb_labels.append(f"{xml_path.name}（{len(lines)} 个框）")
                except Exception as exc:  # noqa: BLE001
                    self.log.emit(f"缩略图生成失败：{img_path}（{exc}）")

        # 保存 classes.txt
        classes_out = (labels_dir if labels_dir else xml_dir) / "classes.txt"
        classes_out.write_text(
            "\n".join(name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])) + "\n",
            encoding="utf-8",
        )
        self.log.emit(f"类别映射（共 {len(class_map)} 类）：{class_map}")
        self.log.emit(f"已保存：{classes_out}")

        # 筛选无标注图片
        unlabeled_count = 0
        if not self.no_unlabeled:
            unlabeled = sorted(
                (img for stem, img in images.items() if stem not in annotated_stems),
                key=lambda p: str(p),
            )
            if unlabeled:
                unlabeled_dir = (
                    Path(self.unlabeled_dir).expanduser()
                    if self.unlabeled_dir else images_dir.parent / "unlabeled"
                )
                unlabeled_dir.mkdir(parents=True, exist_ok=True)
                moved = 0
                for img in unlabeled:
                    dst = unlabeled_dir / img.name
                    if dst.exists():
                        self.log.emit(f"[跳过] 目标已存在：{dst}")
                        continue
                    if self.move_unlabeled:
                        shutil.move(str(img), str(dst))
                    else:
                        shutil.copy2(img, dst)
                    moved += 1
                unlabeled_count = moved
                action = "移动" if self.move_unlabeled else "复制"
                self.log.emit(f"无标注图片 {len(unlabeled)} 张，已{action} {moved} 张到：{unlabeled_dir}")
            else:
                self.log.emit("所有图片都有标注。")

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：转换 {ok} 张标注，失败 {fail} 张。")
        self.finished.emit(ok, fail, unlabeled_count)


class Xml2YoloModule(QtWidgets.QWidget):
    MODULE_TITLE = "标注转换"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把 Pascal VOC / LabelImg 的 XML 标注转成 YOLO txt，\n"
            "并筛选出没有标注的图片到单独文件夹。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # images 目录
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(QtWidgets.QLabel("images 目录:"))
        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText("图片文件夹（必填）")
        img_row.addWidget(self.images_edit, 1)
        img_btn = QtWidgets.QPushButton("浏览")
        img_btn.clicked.connect(self._pick_images)
        img_row.addWidget(img_btn)
        layout.addLayout(img_row)

        # xml 目录
        xml_row = QtWidgets.QHBoxLayout()
        xml_row.addWidget(QtWidgets.QLabel("xml 目录:"))
        self.xml_edit = QtWidgets.QLineEdit()
        self.xml_edit.setPlaceholderText("留空表示与图片同目录")
        xml_row.addWidget(self.xml_edit, 1)
        xml_btn = QtWidgets.QPushButton("浏览")
        xml_btn.clicked.connect(self._pick_xml)
        xml_row.addWidget(xml_btn)
        layout.addLayout(xml_row)

        # labels 目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("labels 目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("留空表示写到 xml 旁边")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 无标注筛选
        unl_row = QtWidgets.QHBoxLayout()
        unl_row.addWidget(QtWidgets.QLabel("无标注输出:"))
        self.unlabeled_edit = QtWidgets.QLineEdit()
        self.unlabeled_edit.setPlaceholderText("留空表示 images 同级下的 unlabeled")
        unl_row.addWidget(self.unlabeled_edit, 1)
        unl_btn = QtWidgets.QPushButton("浏览")
        unl_btn.clicked.connect(self._pick_unlabeled)
        unl_row.addWidget(unl_btn)
        layout.addLayout(unl_row)

        opt_row = QtWidgets.QHBoxLayout()
        self.no_unl_check = QtWidgets.QCheckBox("跳过无标注筛选")
        opt_row.addWidget(self.no_unl_check)
        opt_row.addSpacing(12)
        opt_row.addWidget(QtWidgets.QLabel("处理方式:"))
        self.move_combo = QtWidgets.QComboBox()
        self.move_combo.addItem("复制", False)
        self.move_combo.addItem("移动", True)
        opt_row.addWidget(self.move_combo)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # 类别
        cls_row = QtWidgets.QHBoxLayout()
        cls_row.addWidget(QtWidgets.QLabel("类别 classes:"))
        self.classes_edit = QtWidgets.QLineEdit()
        self.classes_edit.setPlaceholderText("留空自动收集；或填 \"durian,background\"，或 classes.txt 路径")
        cls_row.addWidget(self.classes_edit, 1)
        layout.addLayout(cls_row)

        # 翻页预览（带框标注）
        self.browser = PreviewBrowser(dual=False)
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

    def _pick_images(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 images 目录")
        if path:
            self.images_edit.setText(path)

    def _pick_xml(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 xml 目录")
        if path:
            self.xml_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_unlabeled(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择无标注图片输出目录")
        if path:
            self.unlabeled_edit.setText(path)

    def _restore_config(self):
        """恢复上次保存的路径与参数。"""
        cfg = load_config("xml2yolo")
        self.images_edit.setText(cfg.get("images_dir", ""))
        self.xml_edit.setText(cfg.get("xml_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.unlabeled_edit.setText(cfg.get("unlabeled_dir", ""))
        self.classes_edit.setText(cfg.get("classes", ""))
        self.no_unl_check.setChecked(bool(cfg.get("no_unlabeled", False)))
        move = cfg.get("move", False)
        idx = self.move_combo.findData(move)
        if idx >= 0:
            self.move_combo.setCurrentIndex(idx)

    def _persist_config(self):
        """保存当前路径与参数。"""
        save_config("xml2yolo", {
            "images_dir": self.images_edit.text().strip(),
            "xml_dir": self.xml_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "unlabeled_dir": self.unlabeled_edit.text().strip(),
            "classes": self.classes_edit.text().strip(),
            "move": self.move_combo.currentData(),
            "no_unlabeled": self.no_unl_check.isChecked(),
        })

    def _start(self):
        images_dir = self.images_edit.text().strip()
        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 images 目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.log_view.appendPlainText("开始转换...")
        self.browser.clear()
        self.start_btn.setEnabled(False)

        self._worker = Xml2YoloWorker(
            images_dir=images_dir,
            xml_dir=self.xml_edit.text().strip(),
            labels_dir=self.labels_edit.text().strip(),
            unlabeled_dir=self.unlabeled_edit.text().strip(),
            classes=self.classes_edit.text().strip(),
            move_unlabeled=self.move_combo.currentData(),
            no_unlabeled=self.no_unl_check.isChecked(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.previews.connect(self._on_previews)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_previews(self, items, labels):
        self.browser.set_data(items, labels)

    def _on_finished(self, ok, fail, unlabeled):
        self.start_btn.setEnabled(True)
        msg = f"转换完成：成功 {ok} 张，失败 {fail} 张。"
        if not self.no_unl_check.isChecked():
            msg += f"\n筛选出无标注图片 {unlabeled} 张。"
        QtWidgets.QMessageBox.information(self, "完成", msg)
