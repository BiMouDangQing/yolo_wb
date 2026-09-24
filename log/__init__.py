"""项目日志包：把各功能模块的运行日志写入 log/ 目录下的文件。

每次启动（首次写日志时）创建一个以「日期 + 时间」命名的日志文件，例如：
    log/2026-09-24_143502.log

文件内每条日志带时间戳与模块名，便于定位问题。
"""

import threading
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent

_current_file = None
_lock = threading.Lock()


def _ensure_file():
    global _current_file
    if _current_file is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        _current_file = LOG_DIR / f"{ts}.log"
    return _current_file


def write_log(section, text):
    """把一段日志写入当前会话日志文件。section 为模块名，text 为日志内容。"""
    with _lock:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
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
    """返回当前会话日志文件的路径（尚未写入日志时为 None）。"""
    return str(_current_file) if _current_file else None
