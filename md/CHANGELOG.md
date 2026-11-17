# 改动日志（CHANGELOG）

按时间倒序记录每次改动。格式：日期 + 模块 + 改动内容。

---

## 2026-11-17

### 新增「图片提取」模块

- **前端**：新增标签页「图片提取」（`QT/modules/extract.py`，第 22 个标签页）。
- **命令行**：新增 `tools/extract_images.py`。
- **功能**：把一个多层嵌套文件夹（`A/子文件夹/子文件夹/图片`）里的图片提取到目标文件夹。
  - 递归扫描任意层级子文件夹（可关闭，只扫描顶层）
  - 输出方式：扁平输出（默认）/ 保留子文件夹结构
  - 同名冲突处理：`rename`（自动加序号，默认）/ `skip`（跳过）/ `prefix`（加父文件夹名前缀）
  - 可选同时提取同名标签（`.txt` / `.json` / `.xml` / `.yaml` / `.csv`）
  - 复制（默认）或移动文件
  - 支持结果预览、进度条、历史配置记忆、运行日志落盘
- **挂载**：`QT/qt.py` 的 `MODULES` 列表加入 `ExtractModule`（模块总数 22）。
- **文档**：更新 `README.md`、`md/QT前端使用说明.md`（新增 3.21 节）、`md/项目结构说明.md`；新增 `md/图片提取.md`。

### 修复

- 恢复被误改名的启动 logo：`801.png` → `lg.png`（`QT/qt.py` 中 `LOGO_PATHS` 引用 `lg.png`）。

---

## 2026-10-28

### 图片重命名增强（`QT/modules/rename.py`、`tools/rename.py`）

- 标签同步支持多格式：原仅 `.txt`，现支持 `.txt` / `.json` / `.xml` / `.yaml` / `.yml` / `.csv`。
- 新增「仅标签重命名」模式：图片目录为空或不存在时，直接原地重命名标签目录下的标签文件。
- 起始序号语义保持：填多少第一张就是多少。

---

## 2026-10-21

### 新增「标签可视化」模块

- **前端**：新增标签页「标签可视化」（`QT/modules/visualize.py`，第 21 个标签页）。
- **命令行**：新增 `tools/visualize.py`。
- **功能**：把 YOLO 标签（检测框 / 分割多边形）绘制到图片上，便于肉眼检查标注。
  - 自动识别格式：5 字段 = 检测框，>5 奇数 = 分割多边形
  - 类别名优先读 `classes.txt`，支持中文

---

## 2026-10-18

### 新增功能

- **图片缩放**（`QT/modules/resize.py`、`tools/resize_dataset.py`）：缩放到指定尺寸（最长边），保留 polygon 标签（归一化坐标不变，标签原样复制）。
- **类别合并**（`QT/modules/label_merge.py`、`tools/label_merge.py`）：把类别 C 合并到 B，改框 id、删 C 行、后续 id 自动前移重排。
- **类别重命名**（`QT/modules/label_rename.py`、`tools/label_rename.py`）：只改 `classes.txt` 的类别名，框的 class_id 保持不变。
- **标注转换增加 YOLO 分割（seg）输出格式**：ISAT 的 `segmentation` 多边形直接转多边形，XML 的矩形转 4 顶点矩形。

### 修复

- **数据分析闪退**：CSV 被占用（如 Excel 打开）时写入异常未捕获导致线程崩溃；为 `analysis.py` / `label_check.py` / `miss_label.py` 的 CSV 写入增加异常保护。
- **转换与分析口径不一致**：0 个有效框的标注文件统一判定为「无标注」（`isat2yolo.py` / `xml2yolo.py` / `QT/modules/xml2yolo.py`）。
- **ISAT 转换数据丢失**：优先用 `segmentation` 计算外接框，`bbox` 按 `[x1,y1,x2,y2]` 解析，解析失败不再误判为无标注。
- **类别 id 不稳定**：自动收集类别改为按名称自然排序，稳定 class_id。
- **标签映射支持输入类别名称**（自动读 `classes.txt` 解析为 ID）。
- **分割格式缩略图绘制 bug**：seg 用多边形绘制（`cv2.polylines`）。
