"""一键启动前端界面。

用法（在已激活 diulian 环境的终端中）：
    python run.py

本脚本会把 QT 目录加入 sys.path，然后调用 QT/qt.py 里的 main() 启动界面。
"""

import sys
from pathlib import Path

# 把 QT 目录加入 sys.path，使 qt.py 内部对 qt_binding / store / modules 的 import 生效
sys.path.insert(0, str(Path(__file__).resolve().parent / "QT"))

from qt import main  # noqa: E402


if __name__ == "__main__":
    main()
