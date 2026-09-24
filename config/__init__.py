"""项目配置包：集中管理各功能模块的配置读写与历史记忆。

目录结构：
    config/
      __init__.py      # 本文件：load / save / history 接口
      config.json      # 各模块「当前」配置（section -> dict）
      history/         # 各模块「历史」配置（<section>.json，最近 N 次，按时间倒序）

各模块通过 load(section) / save(section, data) 读写自己的配置，
不同 section 互不干扰（选择文件夹时按模块隔离记忆）。
每次 save 会同时把旧值追加进历史，便于回看「之前几次」的参数。
"""

import json
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_DIR = CONFIG_DIR / "history"
HISTORY_LIMIT = 10  # 每个模块最多保留的历史条数


def _read_all():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_all(data):
    try:
        CONFIG_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def load(section):
    """读取某模块的当前配置，返回 dict；不存在或损坏返回空 dict。"""
    value = _read_all().get(section, {})
    return value if isinstance(value, dict) else {}


def save(section, data):
    """保存某模块的当前配置，并追加一条历史记录（与上次相同则不重复）。"""
    all_data = _read_all()
    prev = all_data.get(section)
    all_data[section] = data
    _write_all(all_data)
    if prev != data:
        _append_history(section, data)


def _history_file(section):
    return HISTORY_DIR / f"{section}.json"


def _read_history(section):
    try:
        data = json.loads(_history_file(section).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _append_history(section, data):
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    entries = _read_history(section)
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
    }
    # 与最近一条相同则不重复记录
    if entries and entries[0].get("data") == data:
        return
    entries.insert(0, entry)
    del entries[HISTORY_LIMIT:]
    try:
        _history_file(section).write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def history(section):
    """返回某模块的历史配置列表（按时间倒序，最多 HISTORY_LIMIT 条）。"""
    return _read_history(section)
