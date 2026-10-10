"""数据集汇总模块：把训练集 / 验证集 / 测试集三个路径下的图片与标签，整理到标准 YOLO 结构。

每个子集路径下图片与标签的常见布局自动识别：
- 混放：train/xxx.jpg + train/xxx.txt（同目录同名）
- 标准：train/images/xxx.jpg + train/labels/xxx.txt

输出结构：
    <输出目录>/
      images/train | val | test
      labels/train | val | test
      data.yaml                       （勾选生成时）
"""

import json
import shutil
from pathlib import Path

from config import load as load_config, save as save_config
from logs import write_log
from modules._history import HistoryBar
from qt_binding import QtCore, QtGui, QtWidgets, Signal

SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def _transfer(src, dst, mode):
    """复制或移动文件到目标位置（自动建目录）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _find_label(p, root):
    """在子集根目录 root 内，为图片 p 定位同名 .txt 标签，找不到返回 None。"""
    txt = p.with_suffix(".txt")
    if txt.is_file():
        return txt
    rel_parts = list(p.relative_to(root).parts)
    if "images" in rel_parts:
        idx = rel_parts.index("images")
        rel_parts[idx] = "labels"
        candidate = root.joinpath(*rel_parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    candidate = root / "labels" / p.relative_to(root).with_suffix(".txt")
    if candidate.is_file():
        return candidate
    return None


def _unique_name(name, used):
    """重名时追加 _2、_3… 后缀，保证目标名唯一。"""
    if name not in used:
        return name
    stem, dot, ext = name.rpartition(".")
    stem = stem or name
    ext = ("." + ext) if dot else ""
    i = 2
    while f"{stem}_{i}{ext}" in used:
        i += 1
    return f"{stem}_{i}{ext}"


class AssembleWorker(QtCore.QThread):
    """后台执行数据集汇总，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)      # (当前进度, 总数)
    finished = Signal(int, int)      # (图片总数, 缺失标签数)

    def __init__(self, train_dir, val_dir, test_dir, output_dir,
                 mode, gen_yaml, nc, names, parent=None):
        super().__init__(parent)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.mode = mode               # copy / move
        self.gen_yaml = gen_yaml
        self.nc = nc
        self.names = names

    def _collect(self, subset, root):
        """扫描一个子集，返回 (图片列表, 缺失标签数)。"""
        root = Path(root).expanduser()
        if not root.is_dir():
            self.log.emit(f"[{subset}] 路径不存在，跳过：{root}")
            return [], 0
        files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        return files, 0

    def run(self):
        subsets = [
            ("train", self.train_dir),
            ("val", self.val_dir),
            ("test", self.test_dir),
        ]
        out = Path(self.output_dir).expanduser()

        # 预扫描，统计总图片数用于进度条
        total = 0
        collected = []
        for subset, d in subsets:
            files, _ = self._collect(subset, d)
            collected.append((subset, d, files))
            total += len(files)

        if total == 0:
            self.log.emit("三个路径下都没有找到图片。")
            self.finished.emit(0, 0)
            return

        self.log.emit(
            f"开始汇总：train/val/test 共 {total} 张图片，"
            f"模式：{'移动' if self.mode == 'move' else '复制'}，输出：{out}"
        )

        done = 0
        missing = 0
        has_labels = False
        for subset, d, files in collected:
            if not files:
                continue
            root = Path(d).expanduser()
            used_names = set()
            for src in files:
                done += 1
                self.progress.emit(done, total)
                name = _unique_name(src.name, used_names)
                used_names.add(name)

                img_dst = out / "images" / subset / name
                _transfer(src, img_dst, self.mode)

                lbl = _find_label(src, root)
                if lbl is not None:
                    has_labels = True
                    lbl_dst = out / "labels" / subset / (name.rsplit(".", 1)[0] + ".txt")
                    _transfer(lbl, lbl_dst, self.mode)
                else:
                    missing += 1
                    self.log.emit(f"[{subset}] 无标签：{src.name}")

        if self.gen_yaml:
            self._write_yaml(out, has_labels)

        self.log.emit(
            f"\n完成：共汇总 {done} 张图片" +
            (f"，缺失标签 {missing} 个。" if missing else "。")
        )
        self.finished.emit(done, missing)

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


class AssembleModule(QtWidgets.QWidget):
    MODULE_TITLE = "数据集汇总"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把训练集 / 验证集 / 测试集三个路径下的图片与同名标签，整理到标准 YOLO 结构：\n"
            "输出目录/images/{train,val,test} 与 输出目录/labels/{train,val,test}。\n"
            "每个子集内图片与标签支持「同目录混放」或「images/labels 分开」两种布局。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 训练集路径
        train_row = QtWidgets.QHBoxLayout()
        train_row.addWidget(QtWidgets.QLabel("训练集路径:"))
        self.train_edit = QtWidgets.QLineEdit()
        self.train_edit.setPlaceholderText("训练集文件夹（必填）")
        train_row.addWidget(self.train_edit, 1)
        train_btn = QtWidgets.QPushButton("浏览")
        train_btn.clicked.connect(self._pick_train)
        train_row.addWidget(train_btn)
        layout.addLayout(train_row)

        # 验证集路径
        val_row = QtWidgets.QHBoxLayout()
        val_row.addWidget(QtWidgets.QLabel("验证集路径:"))
        self.val_edit = QtWidgets.QLineEdit()
        self.val_edit.setPlaceholderText("验证集文件夹（必填）")
        val_row.addWidget(self.val_edit, 1)
        val_btn = QtWidgets.QPushButton("浏览")
        val_btn.clicked.connect(self._pick_val)
        val_row.addWidget(val_btn)
        layout.addLayout(val_row)

        # 测试集路径
        test_row = QtWidgets.QHBoxLayout()
        test_row.addWidget(QtWidgets.QLabel("测试集路径:"))
        self.test_edit = QtWidgets.QLineEdit()
        self.test_edit.setPlaceholderText("测试集文件夹（可选，留空则跳过）")
        test_row.addWidget(self.test_edit, 1)
        test_btn = QtWidgets.QPushButton("浏览")
        test_btn.clicked.connect(self._pick_test)
        test_row.addWidget(test_btn)
        layout.addLayout(test_row)

        # 输出目录
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("输出目录:"))
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("整理后的标准数据集保存位置")
        out_row.addWidget(self.output_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # 选项
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("模式:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("复制（保留原文件）", "copy")
        self.mode_combo.addItem("移动（原文件转移到输出目录）", "move")
        opt_row.addWidget(self.mode_combo)
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

        # 历史配置恢复
        self.history_bar = HistoryBar("assemble", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始汇总")
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

    def _pick_train(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择训练集文件夹")
        if path:
            self.train_edit.setText(path)

    def _pick_val(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择验证集文件夹")
        if path:
            self.val_edit.setText(path)

    def _pick_test(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择测试集文件夹")
        if path:
            self.test_edit.setText(path)

    def _pick_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def _restore_config(self):
        self._apply_config(load_config("assemble"))

    def _apply_config(self, cfg):
        self.train_edit.setText(cfg.get("train_dir", ""))
        self.val_edit.setText(cfg.get("val_dir", ""))
        self.test_edit.setText(cfg.get("test_dir", ""))
        self.output_edit.setText(cfg.get("output_dir", ""))
        self.nc_spin.setValue(int(cfg.get("nc", 1)))
        self.names_edit.setText(cfg.get("names", "class0"))
        self.yaml_check.setChecked(bool(cfg.get("gen_yaml", True)))
        mode = cfg.get("mode", "copy")
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def _persist_config(self):
        save_config("assemble", {
            "train_dir": self.train_edit.text().strip(),
            "val_dir": self.val_edit.text().strip(),
            "test_dir": self.test_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "mode": self.mode_combo.currentData(),
            "gen_yaml": self.yaml_check.isChecked(),
            "nc": self.nc_spin.value(),
            "names": self.names_edit.text(),
        })

    def _start(self):
        train_dir = self.train_edit.text().strip()
        val_dir = self.val_edit.text().strip()
        test_dir = self.test_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not train_dir and not val_dir and not test_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "训练集、验证集、测试集至少填一个。")
            return
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输出目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = AssembleWorker(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            mode=self.mode_combo.currentData(),
            gen_yaml=self.yaml_check.isChecked(),
            nc=self.nc_spin.value(),
            names=self.names_edit.text(),
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

    def _on_finished(self, count, missing):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成", f"汇总完成：共 {count} 张图片" +
            (f"，{missing} 张缺失标签。" if missing else "。")
        )
