"""图片提取模块：把一个多层嵌套文件夹里的所有图片提取到目标文件夹。

支持：
    - 递归扫描任意层级的子文件夹
    - 扁平输出（默认）或保留子文件夹结构
    - 命名冲突处理：rename（自动加序号）/ skip（跳过）/ prefix（加父文件夹名前缀）
    - 可选同时提取同名标签（.txt / .json / .xml / .yaml / .csv）
    - 复制（默认）或移动
"""

import shutil
from pathlib import Path

from config import load as load_config, save as save_config
from logs import write_log
from modules._history import HistoryBar
from modules._preview import PreviewBrowser, make_thumb_bgr
from qt_binding import QtCore, QtGui, QtWidgets, Signal

IMAGE_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF", ".GIF", ".HEIC",
}

# 可同时提取的标签文件格式
LABEL_EXTS = (".txt", ".json", ".xml", ".yaml", ".yml", ".csv")


def find_images(src, recursive=True):
    """扫描源目录下的所有图片，返回排序后的列表。"""
    src = Path(src).expanduser()
    it = src.rglob("*") if recursive else src.glob("*")
    return sorted(
        p for p in it
        if p.is_file() and p.suffix.upper() in IMAGE_EXTS
    )


def find_label_for(img, labels_dir=None):
    """查找图片对应的标签文件，找不到返回 None。

    查找顺序：
        1. 与图片同目录、同名的标签文件（images 与 labels 混放）
        2. labels_dir 下按相对路径同名（去掉 images 层级）的标签
        3. labels_dir 下按文件名同名的标签
    """
    for ext in LABEL_EXTS:
        sibling = img.with_suffix(ext)
        if sibling.is_file():
            return sibling

    if labels_dir is not None:
        labels_dir = Path(labels_dir).expanduser()
        for ext in LABEL_EXTS:
            for parent in list(img.parents):
                try:
                    rel_parent = img.relative_to(parent)
                except ValueError:
                    continue
                cand = labels_dir / rel_parent.with_suffix(ext)
                if cand.is_file():
                    return cand
            cand = labels_dir / (img.stem + ext)
            if cand.is_file():
                return cand
    return None


def unique_name(dst_dir, name, used):
    """在 dst_dir 下为 name 生成不冲突的文件名。"""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = name
    i = 1
    while candidate.lower() in used or (dst_dir / candidate).exists():
        candidate = f"{stem}_{i}{suffix}"
        i += 1
    used.add(candidate.lower())
    return candidate


def resolve_conflict(dst_dir, name, strategy, used, img):
    """按冲突策略生成目标文件名；skip 策略命中冲突时返回 None。"""
    if strategy == "skip":
        if name.lower() in used or (dst_dir / name).exists():
            return None
        used.add(name.lower())
        return name
    if strategy == "prefix":
        parent = img.parent.name
        candidate = f"{parent}_{Path(name).stem}{Path(name).suffix}"
        return unique_name(dst_dir, candidate, used)
    return unique_name(dst_dir, name, used)


class ExtractWorker(QtCore.QThread):
    """后台执行图片提取，避免阻塞界面。"""

    log = Signal(str)
    progress = Signal(int, int)        # (当前进度, 总数)
    previews = Signal(object, object)  # (缩略图列表, 文件名列表)
    finished = Signal(int, int, int)   # (成功数, 跳过数, 失败数)

    def __init__(self, src_dir, out_dir, recursive=True, keep_structure=False,
                 conflict="rename", with_labels=False, labels_dir="",
                 move=False, parent=None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.recursive = recursive
        self.keep_structure = keep_structure
        self.conflict = conflict
        self.with_labels = with_labels
        self.labels_dir = labels_dir
        self.move = move

    def run(self):
        src = Path(self.src_dir).expanduser()
        if not src.is_dir():
            self.log.emit(f"源文件夹不存在：{src}")
            self.finished.emit(0, 0, 0)
            return
        if not str(self.out_dir).strip():
            self.log.emit("请指定目标文件夹。")
            self.finished.emit(0, 0, 0)
            return

        out = Path(self.out_dir).expanduser()
        if out.resolve() == src.resolve():
            self.log.emit("目标文件夹不能与源文件夹相同。")
            self.finished.emit(0, 0, 0)
            return

        images = find_images(src, recursive=self.recursive)
        if not images:
            self.log.emit("源文件夹中没有找到图片。")
            self.finished.emit(0, 0, 0)
            return

        labels_dir = None
        if self.with_labels and str(self.labels_dir).strip():
            labels_dir = Path(self.labels_dir).expanduser()
            if not labels_dir.is_dir():
                self.log.emit(f"标签目录不存在（将不提取标签）：{labels_dir}")
                labels_dir = None

        out.mkdir(parents=True, exist_ok=True)
        action = shutil.move if self.move else shutil.copy2
        self.log.emit(
            f"共找到 {len(images)} 张图片，目标：{out}"
            f"（{'移动' if self.move else '复制'}，"
            f"{'保留结构' if self.keep_structure else '扁平输出'}）"
        )

        used_targets = {}
        ok = skip = fail = 0
        label_ok = label_miss = 0
        thumbs = []
        thumb_labels = []
        total = len(images)

        for i, img in enumerate(images):
            self.progress.emit(i + 1, total)

            if self.keep_structure:
                rel = img.relative_to(src)
                dst_dir = out / rel.parent
                dst_dir.mkdir(parents=True, exist_ok=True)
            else:
                dst_dir = out
            used = used_targets.setdefault(str(dst_dir), set())

            name = resolve_conflict(dst_dir, img.name, self.conflict, used, img)
            if name is None:
                skip += 1
                self.log.emit(f"[{i + 1}/{total}] [跳过] {img.name}（同名已存在）")
                continue

            dst = dst_dir / name
            try:
                action(str(img), str(dst))
            except Exception as exc:  # noqa: BLE001
                fail += 1
                self.log.emit(f"[{i + 1}/{total}] [失败] {img.name}：{exc}")
                continue

            ok += 1
            # 同步标签
            if self.with_labels:
                lbl = find_label_for(img, labels_dir)
                if lbl is not None:
                    lbl_dst = dst.with_suffix(lbl.suffix)
                    try:
                        shutil.copy2(str(lbl), str(lbl_dst))
                        label_ok += 1
                    except Exception as exc:  # noqa: BLE001
                        self.log.emit(f"        [标签失败] {lbl.name}：{exc}")
                else:
                    label_miss += 1

            src_hint = ""
            if self.keep_structure:
                try:
                    src_hint = f"  ←  {img.parent.relative_to(src)}"
                except ValueError:
                    src_hint = ""
            self.log.emit(f"[{i + 1}/{total}] {img.name} → {name}{src_hint}")

            if len(thumbs) < 200:
                try:
                    img_bgr = _imread_unicode(dst)
                    if img_bgr is not None:
                        thumb = make_thumb_bgr(img_bgr)
                        thumbs.append((thumb, thumb))
                        thumb_labels.append(str(dst))
                except Exception:  # noqa: BLE001
                    pass

        if thumbs:
            self.previews.emit(thumbs, thumb_labels)

        summary = f"\n完成：提取 {ok} 张，跳过 {skip} 张，失败 {fail} 张"
        if self.with_labels:
            summary += f"，标签 {label_ok} 个，缺标签 {label_miss} 张"
        self.log.emit(summary + f"\n目标文件夹：{out}")
        self.finished.emit(ok, skip, fail)


def _imread_unicode(path):
    """读取图片（支持中文等非 ASCII 路径）。"""
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class ExtractModule(QtWidgets.QWidget):
    MODULE_TITLE = "图片提取"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "把一个多层嵌套文件夹（A/子文件夹/子文件夹/图片）里的所有图片提取到目标文件夹。\n"
            "默认扁平输出（所有图片铺平到一层），也可保留原有子文件夹结构。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 源文件夹
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(QtWidgets.QLabel("源文件夹:"))
        self.src_edit = QtWidgets.QLineEdit()
        self.src_edit.setPlaceholderText("包含多层子文件夹的文件夹（必填）")
        src_row.addWidget(self.src_edit, 1)
        src_btn = QtWidgets.QPushButton("浏览")
        src_btn.clicked.connect(self._pick_src)
        src_row.addWidget(src_btn)
        layout.addLayout(src_row)

        # 目标文件夹
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("目标文件夹:"))
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("提取到的目标文件夹（必填）")
        out_row.addWidget(self.out_edit, 1)
        out_btn = QtWidgets.QPushButton("浏览")
        out_btn.clicked.connect(self._pick_out)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # 标签目录
        lbl_row = QtWidgets.QHBoxLayout()
        lbl_row.addWidget(QtWidgets.QLabel("标签目录:"))
        self.labels_edit = QtWidgets.QLineEdit()
        self.labels_edit.setPlaceholderText("勾选「同时提取标签」时使用（可选）")
        lbl_row.addWidget(self.labels_edit, 1)
        lbl_btn = QtWidgets.QPushButton("浏览")
        lbl_btn.clicked.connect(self._pick_labels)
        lbl_row.addWidget(lbl_btn)
        layout.addLayout(lbl_row)

        # 参数行 1
        param_row = QtWidgets.QHBoxLayout()
        self.recursive_check = QtWidgets.QCheckBox("递归子文件夹")
        self.recursive_check.setChecked(True)
        param_row.addWidget(self.recursive_check)
        param_row.addSpacing(12)
        self.keep_check = QtWidgets.QCheckBox("保留子文件夹结构")
        self.keep_check.setChecked(False)
        param_row.addWidget(self.keep_check)
        param_row.addSpacing(12)
        self.move_check = QtWidgets.QCheckBox("移动（默认复制）")
        self.move_check.setChecked(False)
        param_row.addWidget(self.move_check)
        param_row.addStretch()
        layout.addLayout(param_row)

        # 参数行 2
        param_row2 = QtWidgets.QHBoxLayout()
        param_row2.addWidget(QtWidgets.QLabel("同名冲突:"))
        self.conflict_combo = QtWidgets.QComboBox()
        self.conflict_combo.addItem("自动加序号（rename）", "rename")
        self.conflict_combo.addItem("跳过（skip）", "skip")
        self.conflict_combo.addItem("加父文件夹名前缀（prefix）", "prefix")
        param_row2.addWidget(self.conflict_combo)
        param_row2.addSpacing(16)
        self.labels_check = QtWidgets.QCheckBox("同时提取标签")
        self.labels_check.setChecked(False)
        param_row2.addWidget(self.labels_check)
        param_row2.addStretch()
        layout.addLayout(param_row2)

        # 翻页预览
        self.browser = PreviewBrowser(dual=False)
        layout.addWidget(self.browser)

        # 历史配置恢复
        self.history_bar = HistoryBar("extract", self._apply_config)
        layout.addWidget(self.history_bar)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始提取")
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

    def _pick_src(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择源文件夹")
        if path:
            self.src_edit.setText(path)

    def _pick_out(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if path:
            self.out_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择标签目录")
        if path:
            self.labels_edit.setText(path)

    def _restore_config(self):
        self._apply_config(load_config("extract"))

    def _apply_config(self, cfg):
        self.src_edit.setText(cfg.get("src_dir", ""))
        self.out_edit.setText(cfg.get("out_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.recursive_check.setChecked(bool(cfg.get("recursive", True)))
        self.keep_check.setChecked(bool(cfg.get("keep_structure", False)))
        self.move_check.setChecked(bool(cfg.get("move", False)))
        self.labels_check.setChecked(bool(cfg.get("with_labels", False)))
        conflict = cfg.get("conflict", "rename")
        idx = self.conflict_combo.findData(conflict)
        if idx >= 0:
            self.conflict_combo.setCurrentIndex(idx)

    def _persist_config(self):
        save_config("extract", {
            "src_dir": self.src_edit.text().strip(),
            "out_dir": self.out_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "recursive": self.recursive_check.isChecked(),
            "keep_structure": self.keep_check.isChecked(),
            "move": self.move_check.isChecked(),
            "with_labels": self.labels_check.isChecked(),
            "conflict": self.conflict_combo.currentData(),
        })

    def _start(self):
        src_dir = self.src_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not src_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择源文件夹。")
            return
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择目标文件夹。")
            return
        if Path(src_dir).expanduser().resolve() == Path(out_dir).expanduser().resolve():
            QtWidgets.QMessageBox.warning(self, "提示", "目标文件夹不能与源文件夹相同。")
            return

        self._persist_config()
        self.log_view.clear()
        self.browser.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.start_btn.setEnabled(False)

        self._worker = ExtractWorker(
            src_dir=src_dir,
            out_dir=out_dir,
            recursive=self.recursive_check.isChecked(),
            keep_structure=self.keep_check.isChecked(),
            conflict=self.conflict_combo.currentData(),
            with_labels=self.labels_check.isChecked(),
            labels_dir=self.labels_edit.text().strip(),
            move=self.move_check.isChecked(),
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

    def _on_finished(self, ok, skip, fail):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成",
            f"提取完成：成功 {ok} 张，跳过 {skip} 张，失败 {fail} 张。"
        )
