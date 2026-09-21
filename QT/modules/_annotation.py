"""标注画布组件：在漏标检测预览中直接画框补标注。

参考 labelImg 的画框交互（鼠标拖拽画矩形框），使用本项目统一的
qt_binding 导入 Qt，适配 PySide6 / PyQt6 / PyQt5。

特性：
- 已有标签框显示为绿色，新画的框显示为红色并标注类别名；
- 鼠标拖拽画框，画完自动加入画布并发射 shapeAdded；
- 右键点击撤销最后一个新框；
- 十字光标提示可画框。
"""

import numpy as np

from qt_binding import QtCore, QtGui, QtWidgets, Signal


class AnnotationCanvas(QtWidgets.QWidget):
    """显示图片并支持鼠标拖拽画矩形框。"""

    shapeAdded = Signal(float, float, float, float)  # (归一化 cx, cy, w, h)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None          # 原始 BGR numpy
        self._pixmap = None       # 原始 QPixmap
        self._scale = 1.0
        self._offset = (0, 0)
        self._existing = []       # 已有框（归一化 xyxy），绿色
        self._new = []            # 新画框 [(cls, cx, cy, w, h)]，红色
        self._names = {}          # cls -> 名称
        self._current_cls = 0
        self._drawing = False
        self._start = None
        self._end = None
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setStyleSheet("background:#202020;")
        self.setCursor(QtCore.Qt.CrossCursor)

    # ---------- 对外接口 ----------
    def set_image(self, bgr, boxes=None, names=None):
        """设置图片（BGR numpy）、已有框（归一化 xyxy）、类别名 dict。"""
        self._img = bgr
        self._existing = list(boxes) if boxes else []
        self._names = names if names else {}
        self._new = []
        self._rebuild_pixmap()
        self.update()

    def set_current_class(self, cls):
        """设置画框使用的类别 id。"""
        self._current_cls = cls

    def get_new_boxes(self):
        """返回新画的框 [(cls, cx, cy, w, h)]。"""
        return list(self._new)

    def undo(self):
        """撤销最后一个新框，成功返回 True。"""
        if self._new:
            self._new.pop()
            self.update()
            return True
        return False

    def clear(self):
        self._img = None
        self._pixmap = None
        self._existing = []
        self._new = []
        self._start = None
        self._end = None
        self._drawing = False
        self.update()

    # ---------- 内部 ----------
    def _rebuild_pixmap(self):
        if self._img is None:
            self._pixmap = None
            return
        rgb = np.ascontiguousarray(self._img[:, :, ::-1])  # BGR -> RGB
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg.copy())
        self._recompute_geometry()

    def _recompute_geometry(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        ww = self.width()
        wh = self.height()
        if ww <= 0 or wh <= 0:
            return
        self._scale = min(ww / pw, wh / ph)
        self._offset = ((ww - pw * self._scale) / 2, (wh - ph * self._scale) / 2)

    def resizeEvent(self, event):
        self._recompute_geometry()
        super().resizeEvent(event)

    def _map_to_norm(self, pos):
        """widget 坐标 -> 图片归一化坐标 (x, y)。"""
        if self._pixmap is None or self._pixmap.isNull():
            return 0.0, 0.0
        px = (pos.x() - self._offset[0]) / self._scale
        py = (pos.y() - self._offset[1]) / self._scale
        nx = px / self._pixmap.width()
        ny = py / self._pixmap.height()
        return min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)

    def _to_widget_rect(self, x1, y1, x2, y2):
        """归一化 xyxy -> widget 坐标 QRectF。"""
        pw = self._pixmap.width() if self._pixmap else 1
        ph = self._pixmap.height() if self._pixmap else 1
        wx1 = self._offset[0] + x1 * pw * self._scale
        wy1 = self._offset[1] + y1 * ph * self._scale
        wx2 = self._offset[0] + x2 * pw * self._scale
        wy2 = self._offset[1] + y2 * ph * self._scale
        return QtCore.QRectF(wx1, wy1, wx2 - wx1, wy2 - wy1)

    def _draw_box(self, painter, rect, color, label=None):
        pen = QtGui.QPen(QtGui.QColor(color), 2)
        painter.setPen(pen)
        painter.drawRect(rect)
        if label:
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.setBrush(QtGui.QColor(color))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label) + 8
            th = fm.height() + 2
            painter.drawRect(int(rect.left()), int(rect.top() - th), tw, th)
            painter.drawText(
                QtCore.QRectF(rect.left(), rect.top() - th, tw, th),
                QtCore.Qt.AlignCenter, label,
            )

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self._pixmap is None or self._pixmap.isNull():
            painter.fillRect(self.rect(), QtGui.QColor("#202020"))
            painter.setPen(QtGui.QColor("#999999"))
            painter.drawText(
                self.rect(), QtCore.Qt.AlignCenter, "暂无图片，请先执行漏标检测"
            )
            painter.end()
            return

        # 画图片（缩放后居中）
        pw = int(self._pixmap.width() * self._scale)
        ph = int(self._pixmap.height() * self._scale)
        painter.drawPixmap(
            QtCore.QRect(int(self._offset[0]), int(self._offset[1]), pw, ph),
            self._pixmap,
        )

        # 画已有框（绿色）
        for box in self._existing:
            self._draw_box(painter, self._to_widget_rect(*box), "#00e676")

        # 画新框（红色 + 类别标签）
        for cls, cx, cy, w, h in self._new:
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            label = self._names.get(cls, str(cls))
            self._draw_box(
                painter, self._to_widget_rect(x1, y1, x2, y2),
                "#ff1744", label,
            )

        # 画正在画的框（黄色）
        if self._drawing and self._start is not None and self._end is not None:
            pen = QtGui.QPen(QtGui.QColor("#ffd600"), 2)
            painter.setPen(pen)
            painter.drawRect(self._to_widget_rect(
                self._start[0], self._start[1], self._end[0], self._end[1]
            ))
        painter.end()

    # ---------- 鼠标事件 ----------
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.undo()
            return
        if ev.button() == QtCore.Qt.LeftButton and self._pixmap is not None:
            self._drawing = True
            self._start = self._map_to_norm(ev.pos())
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._end = self._map_to_norm(ev.pos())
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._drawing:
            self._drawing = False
            self._end = self._map_to_norm(ev.pos())
            x1, y1 = self._start
            x2, y2 = self._end
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            self.update()
            if w > 0.002 and h > 0.002:
                self._new.append((self._current_cls, cx, cy, w, h))
                self.update()
                self.shapeAdded.emit(cx, cy, w, h)
