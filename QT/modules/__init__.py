"""功能模块包。

每个模块都是一个 QWidget 子类，提供 MODULE_TITLE 类属性作为标签页标题。
主程序通过 import 引入模块后，用 MainWindow.add_module() 挂载即可。

新增功能模块的步骤：
1. 在本目录新建一个 .py 文件，定义继承 QWidget 的模块类（带 MODULE_TITLE）；
2. 在 QT/qt.py 的 main() 中 import 该模块并调用 add_module()。
"""
