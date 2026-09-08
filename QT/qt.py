"""图片工具前端主程序。

通过 import 引入功能模块并挂载到标签页。新增功能只需：
1. 在 modules/ 下新建模块（继承 QWidget，提供 MODULE_TITLE 类属性）；
2. 在 main() 里 import 并 add_module() 即可。
"""

import sys

from qt_binding import QtWidgets
from store import ModelStore
from modules.converter import ConverterModule
from modules.dedup import DedupModule
from modules.model_manager import ModelManagerModule
from modules.overexposure import OverexposureModule
from modules.predictor import PredictorModule
from modules.quality import QualityModule
from modules.split import SplitModule
from modules.underexposure import UnderexposureModule
from modules.white_balance import WhiteBalanceModule
from modules.xml2yolo import Xml2YoloModule


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片工具")
        self.resize(760, 640)

        self.store = ModelStore()

        self.tabs = QtWidgets.QTabWidget()
        self.add_module(ConverterModule())
        self.add_module(Xml2YoloModule())
        self.add_module(WhiteBalanceModule())
        self.add_module(DedupModule())
        self.add_module(QualityModule())
        self.add_module(OverexposureModule())
        self.add_module(UnderexposureModule())
        self.add_module(SplitModule())
        self.add_module(ModelManagerModule(self.store))
        self.add_module(PredictorModule(self.store))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tabs)

    def add_module(self, widget):
        """把一个功能模块（QWidget）挂载为一个标签页。"""
        title = getattr(widget, "MODULE_TITLE", widget.__class__.__name__)
        self.tabs.addTab(widget, title)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

