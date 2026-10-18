# yolo_data — YOLO 训练图片数据处理工具

一个基于 PySide6 / PyQt 的多标签页桌面工具，专注于 **YOLO（ultralytics）训练前的图片数据处理**：格式统一、数据清洗、标注转换、数据增强、数据集分析与切分，让原始图片快速变成可直接训练的干净数据集。

> 项目背景：duilian-yolo26（榴莲目标检测）。界面风格为柔和绿色主调。

---

## ✨ 功能特性

程序按推荐处理顺序提供了 20 个标签页：

| # | 功能 | 说明 |
| --- | --- | --- |
| 1 | 图片转换 | HEIC/其他格式 → JPG，支持原地转换删除原图 |
| 2 | 标注转换 | Pascal VOC / LabelImg XML 或 ISAT JSON → YOLO txt（检测框 / 分割多边形），并筛选无标注图片 |
| 3 | 标注质检 | 检测格式错误 / 越界框 / 零负宽高 / 极小框 / 巨框 / 空标签 / 类别 ID 越界 |
| 4 | 漏标检测 | 用模型找「模型框出来但标签没有」的疑似漏标，支持在线补框 |
| 5 | 白平衡 | 灰度世界 / 白斑法 / 完美反射 / 手动增益，统一色温 |
| 6 | 图片去重 | 感知哈希（dHash）找出重复或高度相似图片 |
| 7 | 废图筛选 | 检测模糊 / 过暗 / 过曝图片 |
| 8 | 过曝处理 | 高光压缩、降低亮度 |
| 9 | 过暗处理 | 暗部提亮、CLAHE 增强细节 |
| 10 | 数据增强 | 翻转 / 旋转 / 亮度 / 对比度 / 饱和度 / 色调 / 噪声，几何增强可同步变换 YOLO 标签 |
| 11 | 数据集分析 | 统计各类别框数量、剔除未标注图片、生成分析 CSV |
| 12 | 数据集切分 | 按比例切分 train / val / test，可生成 data.yaml |
| 13 | 背景添加 | 背景图作负样本（按占比自动计算数量 + 随机增强扩充 + 空标签），降低误检 |
| 14 | 图片重命名 | 按序号从指定起点连续重命名图片（可同步重命名同名 YOLO 标签） |
| 15 | 数据集汇总 | 把 train/val/test 三个路径下的图片与标签整理到标准 YOLO 结构 |
| 16 | 标签统一 | 把 YOLO 标签所有框的类别 ID 统一改成指定类别（单类别检测） |
| 17 | 标签映射 | 把指定类别 A 的框改成类别 B（A/B 可填类别名称或 ID），其余类别保持不变 |
| 18 | 类别重命名 | 把类别名称 A 改成 B（只改 classes.txt，框的 class_id 保持不变） |
| 19 | 类别合并 | 把类别 C 合并到 B（改框 id、删 C 行、后续 id 自动前移重排） |
| 20 | 图片缩放 | 把图片缩放到指定尺寸（最长边），生成新数据集并保留 polygon 标签 |

所有支持预览的模块都提供「原图 / 结果」翻页预览；所有批量处理功能页均提供进度条，并会记住上次填写的路径与参数。

> 📌 **配置记忆**：每个功能页的路径与参数按模块隔离保存（互不干扰），并自动保留最近 10 次历史配置；页面顶部提供「历史配置」下拉框，可一键恢复之前某次的参数，存于 `config/`。
> 📌 **运行日志**：每次运行会把日志写入 `logs/` 目录，按「月份 / 日期」组织（如 `logs/2026-10/2026-10-10.log`），便于回溯排查。

---

## 🚀 快速开始

### 环境要求

- Windows（推荐）或其它支持 Qt 的操作系统
- Python 3.8+
- 依赖包：`PySide6`（或 PyQt6 / PyQt5）、`opencv-python`、`numpy`、`Pillow`
- 可选：`pillow-heif`（HEIC/HEIF 图片转换需要）；`ultralytics`（漏标检测功能需要）

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
│       ├── label_check.py  # 标注质检
│       ├── miss_label.py   # 漏标检测
│       ├── white_balance.py# 白平衡
│       ├── dedup.py        # 图片去重
│       ├── quality.py      # 废图筛选
│       ├── overexposure.py # 过曝处理
│       ├── underexposure.py# 过暗处理
│       ├── augment.py      # 数据增强
│       ├── analysis.py     # 数据集分析
│       ├── split.py        # 数据集切分
│       ├── background.py   # 背景添加（负样本）
│       ├── rename.py       # 图片重命名（按序号连续命名）
│       ├── assemble.py     # 数据集汇总（train/val/test 整理为标准结构）
│       ├── label_unify.py  # 标签统一（全部类别改成一类）
│       ├── label_remap.py  # 标签映射（类别 A → 类别 B）
│       ├── label_rename.py # 类别重命名（类别名称 A → B）
│       ├── label_merge.py  # 类别合并（类别 C → B，id 重排）
│       └── resize.py       # 图片缩放（缩放到指定尺寸，保留标签）
├── config/                 # 配置包：各模块配置读写 + 历史记忆
│   ├── __init__.py         # load / save / history
│   ├── config.json         # 各模块「当前」配置（本地路径，不入库）
│   └── history/            # 各模块「历史」配置（最近 N 次）
├── logs/                   # 运行日志包
│   ├── __init__.py         # write_log 写入日志文件
│   └── YYYY-MM/            # 按月份子目录 + 日期文件（如 2026-10/2026-10-10.log，不入库）
├── tools/                  # 命令行脚本（独立可运行，不依赖 Qt）
│   ├── jpg.py / heic2jpg.py# 图片转换
│   ├── xml2yolo.py         # 标注转换（XML → YOLO）
│   ├── isat2yolo.py        # 标注转换（ISAT JSON → YOLO）
│   ├── label_check.py      # 标注质检
│   ├── miss_label.py       # 漏标检测
│   ├── overexposure.py / underexposure.py  # 过曝 / 过暗
│   ├── augment.py          # 数据增强
│   ├── analysis.py         # 数据集分析
│   ├── add_backgrounds.py  # 背景添加（负样本）
│   ├── rename.py           # 图片重命名（按序号连续命名）
│   ├── assemble.py         # 数据集汇总（train/val/test 整理为标准结构）
│   ├── label_unify.py      # 标签统一（全部类别改成一类）
│   ├── label_rename.py     # 类别重命名（类别名称 A → B）
│   ├── label_remap.py      # 标签映射（类别 A → 类别 B）
│   ├── label_merge.py      # 类别合并（类别 C → B，id 重排）
│   └── resize_dataset.py   # 图片缩放（缩放到指定尺寸，保留标签）
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
python tools/isat2yolo.py -i ./images -j ./jsons -l ./labels -c "durian,background"
python tools/isat2yolo.py -i ./images -j ./jsons -l ./labels --out-format seg  # YOLO 分割
python tools/label_check.py -i ./labels -o ./label_issues.csv --nc 2
python tools/miss_label.py -m ./best.pt -i ./images -l ./labels -o ./miss.csv --conf 0.3
python tools/overexposure.py -i ./images -o ./fixed --strength 50
python tools/underexposure.py -i ./images -o ./fixed --strength 50
python tools/augment.py -i ./images -l ./labels -o ./aug --hflip --rot90 --brightness 0.8
python tools/analysis.py -i ./images -l ./labels -o ./analysis.csv --remove --move
python tools/add_backgrounds.py -b ./backgrounds -d ./dataset -r 10
python tools/assemble.py --train ./train --val ./val --test ./test -o ./dataset --nc 1 --names durian
python tools/rename.py -i ./images -o ./renamed --start 1 --width 4 --prefix img_
python tools/label_unify.py -i ./labels --target 0
python tools/label_remap.py -i ./labels --src 0-apple --dst 0
python tools/label_rename.py -i ./labels --src 0-apple --dst apple
python tools/label_merge.py -i ./labels --src 1-LightBruise --dst apple
python tools/resize_dataset.py -i ./dataset/images -o ./dataset_1280 -s 1280
```

---

## 🔄 推荐处理流程

```
格式统一（图片转换）
  → 标注转换（XML → YOLO txt）
  → 标注质检（格式 / 几何硬伤检测）
  → 漏标检测（模型反查疑似漏标）
  → 白平衡（统一色温）
  → 数据清洗（去重 / 剔废图）
  → 过曝 / 过暗校正
  → 数据增强
  → 数据集分析（统计标注 / 剔除未标注 / 生成 CSV）
  → 数据集切分（train / val / test）
  → 数据集汇总（train / val / test 整理为标准结构）
  → 背景添加（背景负样本，降低误检）
  → 图片重命名（按序号统一命名）
  → 标签统一 / 标签映射（类别归一 / 类别转换）
```

---

## 📖 文档

- [QT 前端使用说明](md/QT前端使用说明.md)：各标签页详细用法
- [YOLO 训练图片预处理流程](md/YOLO训练图片预处理流程.md)：训练前预处理思路
- [项目结构说明](md/项目结构说明.md)：代码组织与新增功能约定

---

## 📄 许可

本项目以图片数据处理为主；「漏标检测」功能需加载 YOLO 模型做辅助推理，模型文件需用户自行准备。
