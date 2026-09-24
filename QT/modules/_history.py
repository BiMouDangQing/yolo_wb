"""历史配置恢复组件：下拉选择之前保存的参数并一键恢复。

各功能页通过 HistoryBar(section, apply_cfg) 使用：
- section   : 配置小节名（如 "rename"）
- apply_cfg : 回调函数，接收一个 dict，把历史参数应用回各控件
"""

from config import history as history_config
from qt_binding import QtWidgets


class HistoryBar(QtWidgets.QWidget):
    """一行「历史配置」下拉框 + 恢复按钮。"""

    def __init__(self, section, apply_cfg, parent=None):
        super().__init__(parent)
        self.section = section
        self.apply_cfg = apply_cfg

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("历史配置:"))
        self.combo = QtWidgets.QComboBox()
        row.addWidget(self.combo, 1)
        self.restore_btn = QtWidgets.QPushButton("恢复")
        self.restore_btn.setMinimumWidth(64)
        self.restore_btn.clicked.connect(self._on_restore)
        row.addWidget(self.restore_btn)
        self.refresh()

    def refresh(self):
        """重新加载该模块的历史配置列表。"""
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("— 选择历史配置 —", None)
        for entry in history_config(self.section):
            data = entry.get("data")
            if isinstance(data, dict):
                self.combo.addItem(str(entry.get("time", "?")), dict(data))
        self.combo.blockSignals(False)

    def _on_restore(self):
        data = self.combo.currentData()
        if data:
            self.apply_cfg(data)
