"""通用翻页预览组件（内部使用，不作为标签页模块）。

约定：worker 线程只生成缩略图（numpy RGB 数组），不创建 QPixmap；
UI 线程收到缩略图后在 PreviewBrowser 内部转成 QPixmap 显示，
避免在非 GUI 线程创建 GUI 对象。
"""

import cv2
import numpy as np

from qt_binding import QtCore, QtGui, QtWidgets


def make_thumb_bgr(img_bgr, max_size=640):
    """把 BGR 图缩放到不超过 max_size，返回 RGB numpy 数组（连续内存）。"""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(
            img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return np.ascontiguousarray(img_bgr[:, :, ::-1])  # BGR -> RGB


def make_thumb_rgb(img_pil, max_size=640):
    """把 PIL 图缩放到不超过 max_size，返回 RGB numpy 数组（连续内存）。"""
    img_pil.thumbnail((max_size, max_size))
    return np.ascontiguousarray(img_pil.convert("RGB"))


def _rgb_to_pixmap(arr):
    h, w, ch = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


class PreviewBrowser(QtWidgets.QWidget):
    """翻页浏览一组预览图。

    - dual=True：并排显示两张（如 原图 / 结果）；
    - dual=False：显示单张 + 底部说明文字。
    """

    def __init__(self, dual=True, parent=None):
        super().__init__(parent)
        self.dual = dual
        self._items = []    # QPixmap 或 (QPixmap, QPixmap)
        self._labels = []
        self._index = -1
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        images_row = QtWidgets.QHBoxLayout()
        self.label_a = self._make_label("暂无预览")
        images_row.addWidget(self.label_a, 1)
        if self.dual:
            self.label_b = self._make_label("")
            images_row.addWidget(self.label_b, 1)
        layout.addLayout(images_row, 1)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        nav_row = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev)
        self.page_label = QtWidgets.QLabel("0 / 0")
        self.page_label.setAlignment(QtCore.Qt.AlignCenter)
        self.next_btn = QtWidgets.QPushButton("下一张")
        self.next_btn.clicked.connect(self.next)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.page_label, 1)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

        self._update()

    @staticmethod
    def _make_label(text):
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumHeight(160)
        # 关键：让 label 大小完全由布局决定，忽略 pixmap 的 sizeHint，
        # 避免翻页时因图片尺寸变化把界面越撑越大、变形。
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored,
        )
        label.setStyleSheet("background:#f0f0f0; border:1px solid #ccc;")
        return label

    def set_data(self, items, labels=None):
        """items：dual 时为 [(rgb_a, rgb_b), ...]，single 时为 [rgb, ...]。

        本方法在 UI 线程把 numpy RGB 数组转成 QPixmap 并缓存。
        """
        self._items = []
        for item in items:
            if self.dual:
                self._items.append((_rgb_to_pixmap(item[0]), _rgb_to_pixmap(item[1])))
            else:
                self._items.append(_rgb_to_pixmap(item))
        self._labels = list(labels) if labels else [""] * len(self._items)
        self._index = 0 if self._items else -1
        self._update()

    def clear(self):
        self._items = []
        self._labels = []
        self._index = -1
        self._update()

    def prev(self):
        if self._items and self._index > 0:
            self._index -= 1
            self._update()

    def next(self):
        if self._items and self._index < len(self._items) - 1:
            self._index += 1
            self._update()

    def _update(self):
        n = len(self._items)
        if n == 0:
            self.label_a.setText("暂无预览")
            if self.dual:
                self.label_b.setText("")
            self.info_label.setText("")
            self.page_label.setText("0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        item = self._items[self._index]
        if self.dual:
            self._set_pixmap(self.label_a, item[0])
            self._set_pixmap(self.label_b, item[1])
        else:
            self._set_pixmap(self.label_a, item)
        self.info_label.setText(self._labels[self._index])
        self.page_label.setText(f"{self._index + 1} / {n}")
        self.prev_btn.setEnabled(self._index > 0)
        self.next_btn.setEnabled(self._index < n - 1)

    @staticmethod
    def _set_pixmap(label, pixmap):
        label.setText("")
        size = label.size()
        # 布局未完成或尺寸无效时不设置，避免空/异常尺寸
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = pixmap.scaled(
            size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)
