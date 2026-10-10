"""项目日志包：把各功能模块的运行日志写入 logs/ 目录下的文件。

按「月份 / 日期」组织目录结构：
    logs/
      2026-10/              # 月份目录（YYYY-MM）
        2026-10-10.log      # 日期文件（YYYY-MM-DD.log）

同一天的日志追加到同一个日期文件；每条日志带时间戳与模块名，便于定位问题。
"""

import threading
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent

_current_file = None
_lock = threading.Lock()


def _ensure_file():
    global _current_file
    now = datetime.now()
    month = now.strftime("%Y-%m")
    day = now.strftime("%Y-%m-%d")
    month_dir = LOG_DIR / month
    month_dir.mkdir(parents=True, exist_ok=True)
    _current_file = month_dir / f"{day}.log"
    return _current_file


def write_log(section, text):
    """把一段日志追加写入当前日期的日志文件。section 为模块名，text 为日志内容。"""
    with _lock:
        log_file = _ensure_file()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = text.splitlines() or [""]
        try:
            with open(log_file, "a", encoding="utf-8") as fp:
                for line in lines:
                    fp.write(f"[{stamp}] [{section}] {line}\n")
        except OSError:
            pass


def current_log_file():
    """返回当前日期日志文件的路径（尚未写入日志时为 None）。"""
    return str(_current_file) if _current_file else None
