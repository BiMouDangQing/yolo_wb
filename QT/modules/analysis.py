"""数据集分析模块：统计标注情况、剔除未标注图片、生成分析 CSV。

功能：
1. 扫描 images 目录，按 YOLO 标签（.txt）判断每张图是否已标注；
2. 把未标注图片复制 / 移动到指定文件夹（可选）；
3. 统计每个类别（class）的标注框数量；
4. 生成数据集分析 CSV（摘要 + 类别统计 + 图片明细）。

复用 converter / split 的模式：QThread 后台处理，信号回传日志与统计结果。
"""

import csv
import shutil
from pathlib import Path

from config import load as load_config, save as save_config
from qt_binding import QtCore, QtGui, QtWidgets, Signal

# 可处理的图片格式（与其它模块一致）
SUPPORTED_EXTS = {
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF",
}


def parse_label_file(path):
    """解析一个 YOLO 标签文件，返回 class_id 列表。

    行格式：`class cx cy w h`（归一化坐标）。无效行会被跳过。
    """
    classes = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
    except OSError:
        pass
    return classes


class AnalysisWorker(QtCore.QThread):
    """后台执行数据集分析，避免阻塞界面。"""

    log = Signal(str)
    stats = Signal(object)          # 统计结果 dict
    finished = Signal(int, int, int, int)  # (总图数, 已标注, 未标注, 总框数)

    def __init__(self, images_dir, labels_dir, unlabeled_dir, csv_path,
                 remove_unlabeled, move_mode, class_names, parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.unlabeled_dir = unlabeled_dir
        self.csv_path = csv_path
        self.remove_unlabeled = remove_unlabeled
        self.move_mode = move_mode      # "move" / "copy"
        self.class_names = class_names  # 逗号分隔的类别名（可选）

    def _resolve_class_names(self, labels_dir):
        """解析类别名：优先用户输入，其次 classes.txt，最后为空。"""
        if self.class_names:
            return [n.strip() for n in self.class_names.split(",") if n.strip()]
        classes_txt = labels_dir / "classes.txt"
        if classes_txt.is_file():
            try:
                return [
                    line.strip()
                    for line in classes_txt.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    if line.strip()
                ]
            except OSError:
                pass
        return []

    @staticmethod
    def _name_of(class_names, class_id):
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return str(class_id)

    def run(self):
        images = Path(self.images_dir).expanduser()
        if not images.is_dir():
            self.log.emit(f"images 目录不存在：{images}")
            self.finished.emit(0, 0, 0, 0)
            return

        files = sorted(
            p for p in images.rglob("*")
            if p.is_file() and p.suffix.upper() in SUPPORTED_EXTS
        )
        if not files:
            self.log.emit("images 目录中没有找到图片。")
            self.finished.emit(0, 0, 0, 0)
            return

        labels_dir = Path(self.labels_dir).expanduser() if self.labels_dir else None
        class_names = self._resolve_class_names(labels_dir if labels_dir else images)
        self.log.emit(
            f"共 {len(files)} 张图片，类别名："
            + (", ".join(class_names) if class_names else "未提供（按 class id 显示）")
        )

        class_counts = {}
        total_boxes = 0
        annotated = 0
        unlabeled_list = []
        details = []

        for img in files:
            rel = img.relative_to(images)
            if labels_dir is not None:
                label_path = labels_dir / rel.with_suffix(".txt")
            else:
                label_path = img.with_suffix(".txt")

            boxes = parse_label_file(label_path) if label_path.is_file() else []
            if boxes:
                annotated += 1
                total_boxes += len(boxes)
                for cls in boxes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                details.append((rel.as_posix(), "是", len(boxes)))
            else:
                unlabeled_list.append(img)
                details.append((rel.as_posix(), "否", 0))

        unlabeled = len(unlabeled_list)

        # 剔除未标注图片
        removed = 0
        if self.remove_unlabeled and unlabeled_list:
            unlabeled_dir = (
                Path(self.unlabeled_dir).expanduser()
                if self.unlabeled_dir
                else images.parent / "unlabeled"
            )
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            for img in unlabeled_list:
                dst = unlabeled_dir / img.name
                if dst.resolve() == img.resolve():
                    continue
                try:
                    if self.move_mode == "move":
                        shutil.move(str(img), str(dst))
                    else:
                        shutil.copy2(img, dst)
                    removed += 1
                    self.log.emit(f"[已{'移动' if self.move_mode == 'move' else '复制'}] {img.name} -> {dst}")
                except OSError as exc:
                    self.log.emit(f"[失败] {img.name}：{exc}")
            if removed:
                self.log.emit(f"未标注图片 {unlabeled} 张，已处理 {removed} 张到：{unlabeled_dir}")

        # 生成 CSV
        csv_path = self._write_csv(
            images, labels_dir, class_names, class_counts,
            len(files), annotated, unlabeled, total_boxes, details,
        )

        stats = {
            "total": len(files),
            "annotated": annotated,
            "unlabeled": unlabeled,
            "total_boxes": total_boxes,
            "class_counts": class_counts,
            "class_names": class_names,
            "csv_path": str(csv_path),
        }
        self.stats.emit(stats)
        self.log.emit(f"已生成数据集分析 CSV：{csv_path}")
        self.log.emit(
            f"\n完成：共 {len(files)} 张，已标注 {annotated} 张，"
            f"未标注 {unlabeled} 张，总框数 {total_boxes}。"
        )
        self.finished.emit(len(files), annotated, unlabeled, total_boxes)

    def _write_csv(self, images, labels_dir, class_names, class_counts,
                   total, annotated, unlabeled, total_boxes, details):
        """生成分析 CSV，返回文件路径。"""
        if self.csv_path:
            csv_path = Path(self.csv_path).expanduser()
            if csv_path.suffix.lower() != ".csv":
                csv_path = csv_path / "dataset_analysis.csv"
        else:
            csv_path = images / "dataset_analysis.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # 摘要
            writer.writerow(["指标", "数值"])
            writer.writerow(["图片总数", total])
            writer.writerow(["已标注", annotated])
            writer.writerow(["未标注", unlabeled])
            writer.writerow(["总标注框", total_boxes])
            writer.writerow([])

            # 类别统计
            writer.writerow(["class_id", "类别", "框数量"])
            for cls in sorted(class_counts):
                writer.writerow([
                    cls,
                    self._name_of(class_names, cls),
                    class_counts[cls],
                ])
            writer.writerow([])

            # 图片明细
            writer.writerow(["图片", "是否标注", "框数量"])
            for name, ok, count in details:
                writer.writerow([name, ok, count])

        return csv_path


class AnalysisModule(QtWidgets.QWidget):
    MODULE_TITLE = "数据集分析"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()
        self._restore_config()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "扫描 images 目录，统计每张图片的标注情况与各类别框数量，\n"
            "可选剔除未标注图片，并生成数据集分析 CSV。"
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

        # 未标注输出目录（留空 = 不剔除）
        unl_row = QtWidgets.QHBoxLayout()
        unl_row.addWidget(QtWidgets.QLabel("未标注输出:"))
        self.unlabeled_edit = QtWidgets.QLineEdit()
        self.unlabeled_edit.setPlaceholderText("留空表示不剔除；填目录则把未标注图片移到该目录")
        unl_row.addWidget(self.unlabeled_edit, 1)
        unl_btn = QtWidgets.QPushButton("浏览")
        unl_btn.clicked.connect(self._pick_unlabeled)
        unl_row.addWidget(unl_btn)
        unl_row.addWidget(QtWidgets.QLabel("方式:"))
        self.move_combo = QtWidgets.QComboBox()
        self.move_combo.addItem("复制", "copy")
        self.move_combo.addItem("移动", "move")
        unl_row.addWidget(self.move_combo)
        layout.addLayout(unl_row)

        # CSV 输出
        csv_row = QtWidgets.QHBoxLayout()
        csv_row.addWidget(QtWidgets.QLabel("CSV 输出:"))
        self.csv_edit = QtWidgets.QLineEdit()
        self.csv_edit.setPlaceholderText("留空表示 images 目录下的 dataset_analysis.csv")
        csv_row.addWidget(self.csv_edit, 1)
        csv_btn = QtWidgets.QPushButton("浏览")
        csv_btn.clicked.connect(self._pick_csv)
        csv_row.addWidget(csv_btn)
        layout.addLayout(csv_row)

        # 类别名
        cls_row = QtWidgets.QHBoxLayout()
        cls_row.addWidget(QtWidgets.QLabel("类别名:"))
        self.classes_edit = QtWidgets.QLineEdit()
        self.classes_edit.setPlaceholderText("逗号分隔，如 durian,background；留空自动读 classes.txt")
        cls_row.addWidget(self.classes_edit, 1)
        layout.addLayout(cls_row)

        # 统计表格
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["class_id", "类别", "框数量"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("开始分析")
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

    def _pick_images(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 images 目录")
        if path:
            self.images_edit.setText(path)

    def _pick_labels(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 labels 目录")
        if path:
            self.labels_edit.setText(path)

    def _pick_unlabeled(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择未标注图片输出目录")
        if path:
            self.unlabeled_edit.setText(path)

    def _pick_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择 CSV 保存位置", "", "CSV 文件 (*.csv)"
        )
        if path:
            self.csv_edit.setText(path)

    def _restore_config(self):
        """恢复上次保存的路径与参数。"""
        cfg = load_config("analysis")
        self.images_edit.setText(cfg.get("images_dir", ""))
        self.labels_edit.setText(cfg.get("labels_dir", ""))
        self.unlabeled_edit.setText(cfg.get("unlabeled_dir", ""))
        self.csv_edit.setText(cfg.get("csv_path", ""))
        self.classes_edit.setText(cfg.get("class_names", ""))
        move = cfg.get("move_mode", "copy")
        idx = self.move_combo.findData(move)
        if idx >= 0:
            self.move_combo.setCurrentIndex(idx)

    def _persist_config(self):
        """保存当前路径与参数，下次启动自动恢复。"""
        save_config("analysis", {
            "images_dir": self.images_edit.text().strip(),
            "labels_dir": self.labels_edit.text().strip(),
            "unlabeled_dir": self.unlabeled_edit.text().strip(),
            "csv_path": self.csv_edit.text().strip(),
            "class_names": self.classes_edit.text().strip(),
            "move_mode": self.move_combo.currentData(),
        })

    def _start(self):
        images_dir = self.images_edit.text().strip()
        unlabeled_dir = self.unlabeled_edit.text().strip()
        if not images_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择 images 目录。")
            return

        self._persist_config()
        self.log_view.clear()
        self.table.setRowCount(0)
        self.start_btn.setEnabled(False)

        self._worker = AnalysisWorker(
            images_dir=images_dir,
            labels_dir=self.labels_edit.text().strip(),
            unlabeled_dir=unlabeled_dir,
            csv_path=self.csv_edit.text().strip(),
            remove_unlabeled=bool(unlabeled_dir),
            move_mode=self.move_combo.currentData(),
            class_names=self.classes_edit.text().strip(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.stats.connect(self._on_stats)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_stats(self, stats):
        self.table.setRowCount(0)
        class_counts = stats["class_counts"]
        names = stats["class_names"]
        for cls in sorted(class_counts):
            name = names[cls] if 0 <= cls < len(names) else str(cls)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cls)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(class_counts[cls])))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _on_finished(self, total, annotated, unlabeled, total_boxes):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "完成",
            f"分析完成：共 {total} 张图片\n"
            f"已标注 {annotated} 张，未标注 {unlabeled} 张，总框数 {total_boxes}。",
        )
