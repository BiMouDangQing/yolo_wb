"""数据集切分模块：把图片（及对应 YOLO 标签）随机切分为 train / val / test。

输出结构（可选 labels，可选 data.yaml）：
    <输出目录>/
      images/train | val | test
      labels/train | val | test      （提供 labels 目录时）
      data.yaml                      （勾选生成时）
"""

import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from qt_binding import QtCore, QtGui, QtWidgets, Signal

from modules._preview import PreviewBrowser, make_thumb_rgb

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def _transfer(src, dst, mode):
    """复制或移动文件到目标位置。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


class SplitWorker(QtCore.QThread):
    """后台执行数据集切分，避免阻塞界面。"""

    log = Signal(str)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int)  # (已切分图片数, 缺失标签数)

    def __init__(self, images_dir, labels_dir, output_dir, train_pct, val_pct,
                 mode, seed, gen_yaml, nc, names, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.mode = mode  # copy / move
        self.seed = seed
        self.gen_yaml = gen_yaml
        self.nc = nc
        self.names = names

    def run(self):
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

        # 打乱顺序
        shuffled = files[:]
        if self.seed:
            random.Random(self.seed).shuffle(shuffled)
        else:
            random.shuffle(shuffled)

        # 预览示例图（前 200 张）
        thumbs = []
        thumb_labels = []
        for p in shuffled[:200]:
            try:
                with Image.open(p) as im:
                    thumbs.append(make_thumb_rgb(ImageOps.exif_transpose(im)))
                thumb_labels.append(p.name)
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"缩略图生成失败：{p}（{exc}）")

        n = len(shuffled)
        n_train = int(round(n * self.train_pct / 100))
        n_val = int(round(n * self.val_pct / 100))
        # 修正边界，保证 test 不为负
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        n_test = n - n_train - n_val

        splits = [
            ("train", shuffled[:n_train]),
            ("val", shuffled[n_train:n_train + n_val]),
            ("test", shuffled[n_train + n_val:]),
        ]

        out = Path(self.output_dir).expanduser()
        labels_dir = Path(self.labels_dir).expanduser() if self.labels_dir else None

        self.log.emit(
            f"共 {n} 张图片：train {n_train} / val {n_val} / test {n_test}，"
            f"模式：{'移动' if self.mode == 'move' else '复制'}"
        )

        missing = 0
        for subset, flist in splits:
            img_dst_dir = out / "images" / subset
            lbl_dst_dir = out / "labels" / subset
            for src in flist:
                rel = src.relative_to(images)
                _transfer(src, img_dst_dir / rel, self.mode)

                if labels_dir is not None:
                    lbl_src = labels_dir / rel.with_suffix(".txt")
                    if lbl_src.is_file():
                        _transfer(lbl_src, lbl_dst_dir / rel.with_suffix(".txt"), self.mode)
                    else:
                        missing += 1
                        self.log.emit(f"[无标签] {rel}")

        if self.gen_yaml:
            self._write_yaml(out, labels_dir is not None)

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        self.log.emit(f"\n完成：已切分 {n} 张图片" +
                      (f"，缺失标签 {missing} 个。" if missing else "。"))
        self.finished.emit(n, missing)

    def _write_yaml(self, out, has_labels):
        names = [x.strip() for x in self.names.split(",") if x.strip()]
        if not names:
            names = [f"class{i}" for i in range(self.nc)]
        nc = len(names) if names else self.nc

        lines = [
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
        ]
        if has_labels:
            lines.append("test: images/test")
        lines.append(f"nc: {nc}")
        lines.append(f"names: {json.dumps(names, ensure_ascii=False)}")

        yaml_path = out / "data.yaml"
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log.emit(f"已生成 {yaml_path}")


class SplitModule(QtWidgets.QWidget):
    MODULE_TITLE = "数据集切分"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "按比例把图片随机切分为 train / val / test。\n"
            "提供 labels 目录时，会同步切分同名 .txt 标签（YOLO 格式）。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # images 目录
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(QtWidgets.QLabel("images 目录:"))
        self.images_edit = QtWidgets.QLineEdit()
        self.images_edit.setPlaceholderText("选择图片文件夹（必填）")
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
        self.output_edit.setPlaceholderText("切分结果保存到该目录")
        out_row.addWidget(self.output_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # 比例
        ratio_row = QtWidgets.QHBoxLayout()
        ratio_row.addWidget(QtWidgets.QLabel("train %:"))
        self.train_spin = QtWidgets.QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(80)
        ratio_row.addWidget(self.train_spin)
        ratio_row.addSpacing(12)
        ratio_row.addWidget(QtWidgets.QLabel("val %:"))
        self.val_spin = QtWidgets.QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(10)
        ratio_row.addWidget(self.val_spin)
        ratio_row.addWidget(QtWidgets.QLabel("  (test 自动 = 100 - train - val)"))
        ratio_row.addStretch()
        layout.addLayout(ratio_row)

        # 选项
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("模式:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("复制（保留原文件）", "copy")
        self.mode_combo.addItem("移动（原文件转移到输出目录）", "move")
        opt_row.addWidget(self.mode_combo)
        opt_row.addSpacing(12)
        opt_row.addWidget(QtWidgets.QLabel("随机种子:"))
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("0 表示每次随机，其余数值可复现切分结果")
        opt_row.addWidget(self.seed_spin)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # data.yaml
        yaml_row = QtWidgets.QHBoxLayout()
        self.yaml_check = QtWidgets.QCheckBox("生成 data.yaml")
        self.yaml_check.setChecked(True)
        yaml_row.addWidget(self.yaml_check)
        yaml_row.addWidget(QtWidgets.QLabel("类别数 nc:"))
        self.nc_spin = QtWidgets.QSpinBox()
        self.nc_spin.setRange(1, 1000)
        self.nc_spin.setValue(1)
        yaml_row.addWidget(self.nc_spin)
        yaml_row.addWidget(QtWidgets.QLabel("类别名(逗号分隔):"))
        self.names_edit = QtWidgets.QLineEdit("class0")
        yaml_row.addWidget(self.names_edit, 1)
        layout.addLayout(yaml_row)

        # 翻页预览（示例图片）
        self.browser = PreviewBrowser(dual=False)
        layout.addWidget(self.browser)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始切分")
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
        train_pct = self.train_spin.value()
        val_pct = self.val_spin.value()

        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 images 目录。")
            return
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输出目录。")
            return
        if train_pct + val_pct > 100:
            QtWidgets.QMessageBox.warning(self, "提示", "train + val 比例不能超过 100。")
            return

        self.log_view.clear()
        self.log_view.appendPlainText("开始切分...")
        self.browser.clear()
        self.start_btn.setEnabled(False)

        self._worker = SplitWorker(
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            output_dir=output_dir,
            train_pct=train_pct,
            val_pct=val_pct,
            mode=self.mode_combo.currentData(),
            seed=self.seed_spin.value(),
            gen_yaml=self.yaml_check.isChecked(),
            nc=self.nc_spin.value(),
            names=self.names_edit.text(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.previews.connect(self._on_previews)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_previews(self, items, labels):
        self.browser.set_data(items, labels)

    def _on_finished(self, count, missing):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"切分完成：共 {count} 张图片" +
            (f"，{missing} 张缺失标签。" if missing else "。")
        )
