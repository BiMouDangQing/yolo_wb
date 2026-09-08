"""模型管理模块：导入 / 删除 / 查看 YOLO 模型，记录自动持久化。"""

from qt_binding import QtWidgets


class ModelManagerModule(QtWidgets.QWidget):
    MODULE_TITLE = "模型管理"

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "导入 YOLO 模型（.pt）到模型库，记录会自动保存，下次启动仍可见。\n"
            "「删除」仅从列表中移除，不会删除磁盘上的模型文件。"
        )
        layout.addWidget(hint)

        btn_row = QtWidgets.QHBoxLayout()
        self.import_btn = QtWidgets.QPushButton("导入模型")
        self.import_btn.clicked.connect(self._import_model)
        self.delete_btn = QtWidgets.QPushButton("删除选中")
        self.delete_btn.clicked.connect(self._delete_model)
        self.refresh_btn = QtWidgets.QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["模型名称", "路径", "导入时间"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def _refresh(self):
        self.table.setRowCount(0)
        for m in self.store.list():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(m["name"]))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(m["path"]))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(m["added"]))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _import_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.onnx *.engine);;所有文件 (*)"
        )
        if not path:
            return
        if self.store.add(path):
            QtWidgets.QMessageBox.information(self, "成功", "模型已导入。")
            self._refresh()
        else:
            QtWidgets.QMessageBox.information(self, "提示", "该模型已在库中。")

    def _delete_model(self):
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选中要删除的模型。")
            return
        name = self.table.item(row, 0).text()
        path = self.table.item(row, 1).text()
        ret = QtWidgets.QMessageBox.question(
            self, "确认删除",
            f"确定从模型库中移除「{name}」吗？\n（不会删除磁盘上的模型文件）",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if ret == QtWidgets.QMessageBox.Yes:
            self.store.remove(path)
            self._refresh()
