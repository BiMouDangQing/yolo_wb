"""模型库持久化：记录历史导入的模型，重启程序后仍然保留。

模型记录保存在本文件同级的 models.json 中。
"""

import json
from datetime import datetime
from pathlib import Path

from qt_binding import QtCore, Signal

CONFIG_FILE = Path(__file__).resolve().parent / "models.json"


class ModelStore(QtCore.QObject):
    """管理模型库（导入 / 删除 / 查询），变更时发出 models_changed 信号。"""

    models_changed = Signal()

    def __init__(self, config_file: Path = CONFIG_FILE):
        super().__init__()
        self._config = Path(config_file)
        self._models = self._load()

    def _load(self):
        try:
            data = json.loads(self._config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        models = []
        for item in data:
            try:
                path = Path(item["path"])
            except (KeyError, TypeError):
                continue
            if path.is_file():
                models.append({
                    "name": item.get("name", path.name),
                    "path": str(path),
                    "added": item.get("added", ""),
                })
        return models

    def _save(self):
        try:
            self._config.write_text(
                json.dumps(self._models, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def list(self):
        """返回当前模型列表的副本。"""
        return [dict(m) for m in self._models]

    def add(self, path) -> bool:
        """按路径导入模型，已存在则返回 False。"""
        path = str(Path(path).resolve())
        for m in self._models:
            if m["path"] == path:
                return False
        self._models.append({
            "name": Path(path).name,
            "path": path,
            "added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._save()
        self.models_changed.emit()
        return True

    def remove(self, path) -> bool:
        """按路径从库中移除模型，成功返回 True。"""
        path = str(Path(path).resolve())
        new_models = [m for m in self._models if m["path"] != path]
        if len(new_models) == len(self._models):
            return False
        self._models = new_models
        self._save()
        self.models_changed.emit()
        return True
