"""模型预测模块：使用导入的 YOLO 模型预测单张图片或整个文件夹。"""

from pathlib import Path

import numpy as np

from qt_binding import QtCore, QtGui, QtWidgets, Signal


def _bgr_to_pixmap(img_bgr):
    """把 BGR numpy 数组转成 QPixmap。"""
    arr = np.ascontiguousarray(img_bgr[:, :, ::-1])  # BGR -> RGB
    h, w, ch = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


class PredictWorker(QtCore.QThread):
    """后台执行模型预测，避免阻塞界面。"""

    log = Signal(str)
    image = Signal(object)  # QPixmap
    finished = Signal(int)   # 处理的图片数

    def __init__(self, model_path, source, conf, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.source = source
        self.conf = conf

    def run(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"无法加载 ultralytics：{exc}")
            self.finished.emit(0)
            return

        self.log.emit(f"正在加载模型：{self.model_path}")
        try:
            model = YOLO(self.model_path)
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"模型加载失败：{exc}")
            self.finished.emit(0)
            return

        self.log.emit("开始预测...")
        try:
            results = model(self.source, conf=self.conf, verbose=False)
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"预测失败：{exc}")
            self.finished.emit(0)
            return

        if not isinstance(results, list):
            results = [results]

        total = 0
        for i, r in enumerate(results):
            img_name = Path(r.path).name if getattr(r, "path", None) else f"第 {i + 1} 张"
            boxes = getattr(r, "boxes", None)
            n = len(boxes) if boxes is not None else 0
            total += n
            self.log.emit(f"[{img_name}] 检测到 {n} 个目标")
            if boxes is not None:
                names = getattr(r, "names", {})
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    self.log.emit(f"    {label}  置信度 {conf:.2f}  框 {xyxy}")

            if i == 0:
                try:
                    self.image.emit(_bgr_to_pixmap(r.plot()))
                except Exception as exc:  # noqa: BLE001
                    self.log.emit(f"预览生成失败：{exc}")

        self.log.emit(f"\n预测完成：共 {len(results)} 张图片，检测到 {total} 个目标。")
        self.finished.emit(len(results))


class PredictorModule(QtWidgets.QWidget):
    MODULE_TITLE = "模型预测"

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self._worker = None
        self._build_ui()
        self._refresh_models()
        self.store.models_changed.connect(self._refresh_models)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 模型选择
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("模型:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(280)
        model_row.addWidget(self.model_combo, 1)
        browse_btn = QtWidgets.QPushButton("导入...")
        browse_btn.clicked.connect(self._import_model)
        model_row.addWidget(browse_btn)
        layout.addLayout(model_row)

        # 输入类型
        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel("输入:"))
        self.single_radio = QtWidgets.QRadioButton("单张图片")
        self.single_radio.setChecked(True)
        self.folder_radio = QtWidgets.QRadioButton("文件夹")
        self.single_radio.toggled.connect(self._on_type_changed)
        type_row.addWidget(self.single_radio)
        type_row.addWidget(self.folder_radio)
        type_row.addStretch()
        layout.addLayout(type_row)

        # 输入路径
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel("路径:"))
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("选择图片文件或文件夹")
        path_row.addWidget(self.path_edit, 1)
        self.path_btn = QtWidgets.QPushButton("浏览")
        self.path_btn.clicked.connect(self._pick_path)
        path_row.addWidget(self.path_btn)
        layout.addLayout(path_row)

        # 选项
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("置信度阈值:"))
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        opt_row.addWidget(self.conf_spin)
        opt_row.addStretch()
        self.start_btn = QtWidgets.QPushButton("开始预测")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._start)
        opt_row.addWidget(self.start_btn)
        layout.addLayout(opt_row)

        # 预览
        self.image_label = QtWidgets.QLabel("预测结果预览")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(240)
        self.image_label.setStyleSheet("background:#f0f0f0; border:1px solid #ccc;")
        layout.addWidget(self.image_label, 1)

        # 日志
        layout.addWidget(QtWidgets.QLabel("结果:"))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

    def _refresh_models(self):
        current = self.model_combo.currentData()
        self.model_combo.clear()
        for m in self.store.list():
            self.model_combo.addItem(m["name"], m["path"])
        if current:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    def _import_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.onnx *.engine);;所有文件 (*)"
        )
        if not path:
            return
        self.store.add(path)
        idx = self.model_combo.findData(str(Path(path).resolve()))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _on_type_changed(self, checked):
        self.path_edit.clear()

    def _pick_path(self):
        if self.folder_radio.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择图片", "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.heic *.heif);;所有文件 (*)"
            )
        if path:
            self.path_edit.setText(path)

    def _start(self):
        model_path = self.model_combo.currentData()
        source = self.path_edit.text().strip()

        if not model_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在【模型管理】页导入模型，或点击「导入...」。")
            return
        if not source:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输入图片或文件夹。")
            return

        self.log_view.clear()
        self.image_label.setText("预测中...")
        self.image_label.setPixmap(QtGui.QPixmap())
        self.start_btn.setEnabled(False)

        self._worker = PredictWorker(model_path, source, self.conf_spin.value())
        self._worker.log.connect(self._append_log)
        self._worker.image.connect(self._show_image)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _show_image(self, pixmap):
        self.image_label.setText("")
        scaled = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _on_finished(self, count):
        self.start_btn.setEnabled(True)
