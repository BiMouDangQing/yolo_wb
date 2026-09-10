# yolo_data — YOLO 训练图片数据处理工具

一个基于 PySide6 / PyQt 的多标签页桌面工具，专注于 **YOLO（ultralytics）训练前的图片数据处理**：格式统一、数据清洗、标注转换、数据增强、数据集分析与切分，让原始图片快速变成可直接训练的干净数据集。

> 项目背景：duilian-yolo26（榴莲目标检测）。界面风格为柔和绿色主调。

---

## ✨ 功能特性

程序按推荐处理顺序提供了 10 个标签页：

| # | 功能 | 说明 |
| --- | --- | --- |
| 1 | 图片转换 | HEIC/其他格式 → JPG，支持原地转换删除原图 |
| 2 | 标注转换 | Pascal VOC / LabelImg XML → YOLO txt，并筛选无标注图片 |
| 3 | 白平衡 | 灰度世界 / 白斑法 / 完美反射 / 手动增益，统一色温 |
| 4 | 图片去重 | 感知哈希（dHash）找出重复或高度相似图片 |
| 5 | 废图筛选 | 检测模糊 / 过暗 / 过曝图片 |
| 6 | 过曝处理 | 高光压缩、降低亮度 |
| 7 | 过暗处理 | 暗部提亮、CLAHE 增强细节 |
| 8 | 数据增强 | 翻转 / 旋转 / 亮度 / 对比度 / 饱和度 / 色调 / 噪声，几何增强可同步变换 YOLO 标签 |
| 9 | 数据集分析 | 统计各类别框数量、剔除未标注图片、生成分析 CSV |
| 10 | 数据集切分 | 按比例切分 train / val / test，可生成 data.yaml |

所有支持预览的模块都提供「原图 / 结果」翻页预览。

---

## 🚀 快速开始

### 环境要求

- Windows（推荐）或其它支持 Qt 的操作系统
- Python 3.8+
- 依赖包：`PySide6`（或 PyQt6 / PyQt5）、`opencv-python`、`numpy`、`Pillow`
- 可选：`pillow-heif`（HEIC/HEIF 图片转换需要）

### 安装依赖

```bash
pip install PySide6 opencv-python numpy Pillow
# 需要 HEIC 转换时：
pip install pillow-heif
```

### 启动

**方式一（推荐）**：双击项目根目录的 `yolo_data.bat`，自动激活环境并启动界面。

**方式二**：在终端中运行：

```bash
python run.py
```

---

## 📁 目录结构

```
yolo_data/
├── run.py                  # 启动入口
├── yolo_data.bat           # Windows 双击启动脚本
├── QT/                     # 前端
│   ├── qt.py               # 主程序：挂载各功能模块为标签页
│   ├── qt_binding.py       # Qt 绑定（PySide6 / PyQt6 / PyQt5 回退）
│   └── modules/            # 每个功能一个模块（算法 + 后台线程 + 页面）
│       ├── _preview.py     # 通用翻页预览组件
│       ├── converter.py    # 图片转换
│       ├── xml2yolo.py     # 标注转换
│       ├── white_balance.py# 白平衡
│       ├── dedup.py        # 图片去重
│       ├── quality.py      # 废图筛选
│       ├── overexposure.py # 过曝处理
│       ├── underexposure.py# 过暗处理
│       ├── augment.py      # 数据增强
│       ├── analysis.py     # 数据集分析
│       └── split.py        # 数据集切分
├── tools/                  # 命令行脚本（独立可运行，不依赖 Qt）
│   ├── jpg.py / heic2jpg.py# 图片转换
│   ├── xml2yolo.py         # 标注转换
│   ├── overexposure.py / underexposure.py  # 过曝 / 过暗
│   ├── augment.py          # 数据增强
│   └── analysis.py         # 数据集分析
└── md/                     # 使用与结构文档
    ├── QT前端使用说明.md
    ├── YOLO训练图片预处理流程.md
    └── 项目结构说明.md
```

---

## 🛠️ 命令行工具

除图形界面外，`tools/` 下提供独立的命令行脚本，便于批处理或脚本化：

```bash
python tools/jpg.py -i ./images -o ./images_jpg
python tools/heic2jpg.py -i ./photos -o ./photos_jpg --overwrite
python tools/xml2yolo.py -i ./images -x ./xml -l ./labels
python tools/overexposure.py -i ./images -o ./fixed --strength 50
python tools/underexposure.py -i ./images -o ./fixed --strength 50
python tools/augment.py -i ./images -l ./labels -o ./aug --hflip --rot90 --brightness 0.8
python tools/analysis.py -i ./images -l ./labels -o ./analysis.csv --remove --move
```

---

## 🔄 推荐处理流程

```
格式统一（图片转换）
  → 标注转换（XML → YOLO txt）
  → 白平衡（统一色温）
  → 数据清洗（去重 / 剔废图）
  → 过曝 / 过暗校正
  → 数据增强
  → 数据集分析（统计标注 / 剔除未标注 / 生成 CSV）
  → 数据集切分（train / val / test）
```

---

## 📖 文档

- [QT 前端使用说明](md/QT前端使用说明.md)：各标签页详细用法
- [YOLO 训练图片预处理流程](md/YOLO训练图片预处理流程.md)：训练前预处理思路
- [项目结构说明](md/项目结构说明.md)：代码组织与新增功能约定

---

## 📄 许可

本项目仅用于图片数据处理，不包含模型训练与推理功能。
