"""Qt 绑定选择：优先 PySide6，其次 PyQt6，最后 PyQt5。

所有模块统一从这里导入 Qt，避免重复写回退逻辑。
"""

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    Signal = QtCore.Signal
    Slot = QtCore.Slot
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets

        Signal = QtCore.pyqtSignal
        Slot = QtCore.pyqtSlot
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets

            Signal = QtCore.pyqtSignal
            Slot = QtCore.pyqtSlot
        except ImportError:
            raise SystemExit(
                "未找到 Qt 绑定库，请先安装其中之一：\n"
                "  pip install PySide6\n"
                "  # 或 pip install PyQt6 / PyQt5"
            )
