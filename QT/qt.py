"""图片工具前端主程序。

通过 import 引入功能模块并挂载到标签页。新增功能只需：
1. 在 modules/ 下新建模块（继承 QWidget，提供 MODULE_TITLE 类属性）；
2. 在下方 MODULES 列表里 import 并加入即可（顺序即标签页顺序）。
"""

import sys
from pathlib import Path

from qt_binding import QtCore, QtGui, QtWidgets
from modules.augment import AugmentModule
from modules.converter import ConverterModule
from modules.dedup import DedupModule
from modules.overexposure import OverexposureModule
from modules.quality import QualityModule
from modules.split import SplitModule
from modules.underexposure import UnderexposureModule
from modules.white_balance import WhiteBalanceModule
from modules.xml2yolo import Xml2YoloModule

# 项目根目录下的 logo 图片（两个都是 logo）
LOGO_PATHS = [
    Path(__file__).resolve().parent.parent / "lg.png",
    Path(__file__).resolve().parent.parent / "log.png",
]

# 全局样式：柔和绿色主调 + 少量红色点缀
STYLESHEET = """
QWidget {
    background-color: #f2f8ec;
    color: #3b5236;
    font-size: 13px;
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
}
QLabel {
    background: transparent;
    color: #3b5236;
}
QTabWidget::pane {
    border: 1px solid #d6e4cd;
    background: #ffffff;
    border-radius: 6px;
}
QTabBar::tab {
    background: #e4f0dc;
    color: #5f7d56;
    padding: 8px 18px;
    border: 1px solid #d6e4cd;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #7cc35e, stop:1 #58a93e);
    color: #ffffff;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #d0e8c2;
    color: #3f7a2c;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #7cc35e, stop:1 #58a93e);
    color: #ffffff;
    border: none;
    padding: 6px 16px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #8fd071, stop:1 #63b449);
}
QPushButton:pressed {
    background: #3d7a27;
}
QPushButton:disabled {
    background: #cfe3c4;
    color: #ffffff;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #d5dbe0;
    border-radius: 3px;
    padding: 4px 6px;
    color: #3b5236;
    selection-background-color: #8ccf6b;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #58a93e;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #d6e4cd;
    selection-background-color: #d0e8c2;
    selection-color: #35563a;
}
QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #d5dbe0;
    border-radius: 3px;
    color: #3b5236;
}
QTableWidget {
    background: #ffffff;
    gridline-color: #e0e4e8;
    alternate-background-color: #f2f8ec;
}
QHeaderView::section {
    background: #58a93e;
    color: #ffffff;
    padding: 6px;
    border: none;
}
QCheckBox, QRadioButton {
    background: transparent;
    color: #3b5236;
    spacing: 4px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #58a93e;
    border: 1px solid #58a93e;
    border-radius: 3px;
}
QGroupBox {
    border: 1px solid #d6e4cd;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #35563a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QScrollBar:vertical {
    border: none;
    background: #eef5e9;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #b8d6a8;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #8cc06f;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    border: none;
    background: #eef5e9;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #b8d6a8;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #8cc06f;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QToolTip {
    background: #35563a;
    color: #ffffff;
    border: none;
    padding: 4px 6px;
}
"""


# 功能模块清单（顺序即标签页顺序，新增功能在此加一行即可）
MODULES = [
    ConverterModule,
    Xml2YoloModule,
    WhiteBalanceModule,
    DedupModule,
    QualityModule,
    OverexposureModule,
    UnderexposureModule,
    AugmentModule,
    SplitModule,
]


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("果蔬图片工具")
        self.resize(820, 740)
        self._set_window_icon()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        for cls in MODULES:
            self.add_module(cls())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(self._build_header())
        layout.addWidget(self.tabs)

    def _set_window_icon(self):
        """设置窗口图标（取第一个存在的 logo）。"""
        for path in LOGO_PATHS:
            if path.is_file():
                self.setWindowIcon(QtGui.QIcon(str(path)))
                return

    def _logo_pixmaps(self, height=44):
        """返回可用的 logo 缩略图（QPixmap 列表）。"""
        pixmaps = []
        for path in LOGO_PATHS:
            if path.is_file():
                pixmap = QtGui.QPixmap(str(path))
                if not pixmap.isNull():
                    pixmaps.append(
                        pixmap.scaledToHeight(height, QtCore.Qt.SmoothTransformation)
                    )
        return pixmaps

    def _build_header(self):
        """顶部：logo + 主标语 + 副标语。"""
        header = QtWidgets.QHBoxLayout()
        for pixmap in self._logo_pixmaps():
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            header.addWidget(label)
            header.addSpacing(10)

        text_col = QtWidgets.QVBoxLayout()
        slogan = QtWidgets.QLabel("🌱 为世界分选好果蔬，与世界共享好成果")
        slogan.setStyleSheet(
            "font-size:17px; font-weight:bold; color:#3f7a2c;"
        )
        sub = QtWidgets.QLabel("🍃 绿色 · 新鲜 · 高效 —— 图片数据处理工具")
        sub.setStyleSheet("font-size:12px; color:#7ba16e;")
        text_col.addWidget(slogan)
        text_col.addWidget(sub)
        header.addLayout(text_col)
        header.addStretch()
        return header

    def add_module(self, widget):
        """把一个功能模块（QWidget）挂载为一个标签页。"""
        title = getattr(widget, "MODULE_TITLE", widget.__class__.__name__)
        self.tabs.addTab(widget, title)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

