"""全局配置持久化：记住各功能页上次填写的路径与参数。

配置保存在本文件同级的 config.json 中。
各模块通过 load(section) / save(section, data) 读写自己的配置。
"""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def load(section):
    """读取某个模块（section）的配置，返回 dict；不存在或损坏返回空 dict。"""
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    value = data.get(section, {})
    return value if isinstance(value, dict) else {}


def save(section, data):
    """保存某个模块（section）的配置（dict），不影响其它 section。"""
    try:
        all_data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        all_data = {}
    if not isinstance(all_data, dict):
        all_data = {}
    all_data[section] = data
    try:
        CONFIG_FILE.write_text(
            json.dumps(all_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
