# Auto YOLO Train Pipeline

YOLOv5 / Ultralytics 自动化训练管道 —— 一键完成数据准备、版本管理、权重下载和模型训练，支持 **YOLOv5 体系（v5.0 / v6.0 / v7.0）** 和 **Ultralytics 体系（YOLOv8 / v10 / v11）**。

---

## 功能特性

- **全自动流程** — 数据准备 → 版本代码同步 → 权重下载 → 模型训练，一行命令跑完
- **双体系支持** — YOLOv5 体系（yolov5-5.0/yolov5-6.0/yolov5-7.0，有子版本）和 Ultralytics 体系（yolov8/yolov10/yolo11，无子版本）通过 `--yolo-version` 参数无缝切换
- **策略模式** — 6 个版本共享同一套抽象接口，各版本差异由策略类封装
- **智能缓存** — YOLOv5 代码库通过 git shallow clone 自动缓存，本地优先、缓存兜底
- **权重管理** — YOLOv5 体系手动下载权重（带进度条），Ultralytics 体系由包自动管理
- **数据预处理** — 支持 YOLO TXT 和 VOC XML 两种标注格式，自动划分训练/验证/测试集，双体系共用
- **配置冻结** — 运行时配置不可变，杜绝意外修改导致的隐式 bug
- **优雅退出** — 注册信号处理器，`Ctrl+C` 安全终止训练
- **双环境支持** — 同时提供 Conda 和 uv 两种环境管理方案

---

## 项目结构

```
auto_yolov5_train/
├── yolo_train_pipeline.py          # 主入口脚本（支持 --yolo-version 参数）
├── pyproject.toml              # uv 项目配置与依赖声明
├── .python-version             # Python 版本锁定（3.10）
├── uv.lock                     # uv 依赖锁定文件（可复现安装）
│
├── scripts/
│   ├── config.py               # 双体系配置：YOLOv5Config + UltralyticsConfig
│   ├── data.py                 # 数据准备（划分数据集、格式转换、生成 data.yaml）
│   ├── trainer.py              # 训练执行（双体系分发 + subprocess + 信号处理）
│   ├── version_manager.py      # YOLOv5 版本代码缓存 + 预训练权重下载
│   ├── ultralytics_manager.py  # Ultralytics 包与环境管理
│   └── strategies/
│       ├── __init__.py         # 策略注册（STRATEGY_MAP）
│       ├── base.py             # VersionStrategy 抽象基类
│       ├── v5.py               # v5.0 策略（hyp.yaml + 命令行标志）
│       ├── v6plus.py           # v6.0+/v7.0 策略（全命令行参数）
│       └── ultralytics.py      # Ultralytics 策略（yolo CLI key=value）
│
├── data/                       # 数据目录（双体系共用）
│   └── .gitkeep                # 占位文件
│   └── raw/                    # 用户原始数据
│       ├── images/             # 图片文件（.jpg / .png）
│       └── labels/             # 标注文件（YOLO .txt 或 VOC .xml）
│
├── project/                    # [运行中生成]训练输出（双体系共用）
│   ├── train.txt / val.txt     # 数据集划分文件
│   ├── data.yaml               # YOLO 数据配置
│   ├── hyp.custom.yaml         # 自定义超参文件（v5.0 生成）
│   └── train_results/          # 训练结果（权重、日志、图表）
│
├── weights/                    # [YOLOv5 体系]预训练权重缓存
│
├── training.log                # [运行中生成]管道运行日志
```

---

## 前置条件

| 依赖 | 说明 |
|------|------|
| Python 3.10 ~ 3.12 | 建议 3.10 |
| Git | YOLOv5 版本代码通过 git clone 自动下载 |
| NVIDIA GPU + CUDA（可选） | 训练 GPU 加速，CPU 亦可运行但极慢 |

---

## 安装

### 方式一：uv（推荐）

```bash
# 安装 uv（如尚未安装）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 创建虚拟环境 + 安装基础依赖
uv sync

# 安装 YOLOv5 训练依赖（如需在本机执行训练）
uv sync --group train

# 安装 Ultralytics 依赖（如需训练 YOLOv8/v10/v11）
uv sync --group ultralytics

# 激活环境
.venv\Scripts\activate
```

> **关于 PyTorch CUDA**：`pyproject.toml` 已配置 CUDA 12.1 索引，`uv sync --group train` 会自动安装 CUDA 版 PyTorch。如需其他 CUDA 版本，请手动安装。

### 方式二：Conda（复现性好）

```bash
# 从 environment.yml 创建环境（自动安装所有依赖）
conda env create -f environment.yml
conda activate auto_yolov5_train
```

> **关于 CUDA / Ultralytics**：`environment.yml` 默认安装 CPU 版 PyTorch。如需 GPU 训练或 Ultralytics（YOLOv8/v10/v11），激活环境后按需安装：
> ```bash
> # CUDA 版 PyTorch（以 CUDA 12.1 为例）
> pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
>
> # Ultralytics
> pip install ultralytics>=8.0.0
> ```

---

## 配置

支持两套配置类，实例化后自动冻结，运行时不可修改。

### YOLOv5 体系配置

所有 YOLOv5 训练参数集中在 [scripts/config.py](file:///d:/auto_yolov5_train/scripts/config.py)。

| 配置模块 | 字段 | 默认值 | 说明 |
|---------|------|--------|------|
| 基础 | `YOLO_VERSION` | `"5.0"` | 可选 `"5.0"` `"6.0"` `"7.0"` |
| 训练 | `batch_size` | `16` | RTX 4060 8GB 建议 16~24 |
| 训练 | `epochs` | `120` | 训练总轮数 |
| 训练 | `img_size` | `640` | 输入图像尺寸（像素） |
| 训练 | `device` | `"0"` | GPU ID 或 `"cpu"` |
| 训练 | `workers` | `12` | 数据加载进程数 |
| 训练 | `model` | `"yolov5s.yaml"` | 模型配置文件 |
| 优化器 | `optimizer` | `"SGD"` | v6.0+ 支持 `SGD` `Adam` `AdamW` |

### Ultralytics 体系配置

所有 Ultralytics 训练参数集中在 [scripts/config.py](file:///d:/auto_yolov5_train/scripts/config.py)（`UltralyticsConfig` 类）。参数命名与 Ultralytics 官方 CLI 一致。

| 配置模块 | 字段 | 默认值 | 说明 |
|---------|------|--------|------|
| 基础 | `MODEL` | `"yolov8"` | 可选 `"yolov8"` `"yolov10"` `"yolo11"`（模型名即标识，无子版本） |
| 训练 | `model` | `"yolov8n.pt"` | 模型文件名，自动依据 MODEL 生成 |
| 训练 | `batch` | `16` | YOLOv5 的 `batch_size` |
| 训练 | `epochs` | `120` | 训练总轮数 |
| 训练 | `imgsz` | `640` | YOLOv5 的 `img_size` |
| 训练 | `device` | `"0"` | GPU ID 或 `"cpu"` |
| 训练 | `workers` | `12` | 数据加载进程数 |
| 训练 | `optimizer` | `"SGD"` | 支持 `SGD` `Adam` `AdamW` `auto` |
| 训练 | `amp` | `True` | 自动混合精度 |
| 增强 | `auto_augment` | `"randaugment"` | Ultralytics 特有 |
| 增强 | `erasing` | `0.4` | Ultralytics 特有 |

### 修改配置

在 `main()` 中实例化时传入参数即可修改：

```python
# YOLOv5 体系
cfg = YOLOv5Config(version="7.0")
cfg = YOLOv5Config(version="6.0", model="yolov5m.yaml")

# Ultralytics 体系
cfg = UltralyticsConfig(model="yolov8")
cfg = UltralyticsConfig(model="yolo11")
```

如需持久化修改，直接编辑对应配置文件的默认值。

---

## 数据准备

### 目录结构

请将训练数据放入 `data/raw/` 目录：

```
data/raw/
├── images/          # 图片文件（.jpg / .jpeg / .png / .bmp）
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/          # 标注文件（YOLO .txt 或 VOC .xml）
    ├── img_001.txt      # YOLO 格式：class_id x_center y_center width height
    ├── img_002.xml      # VOC 格式：自动转换为 YOLO
    └── ...
```

### 标注格式

- **YOLO TXT**（推荐）：每行一个目标，格式为 `class_id x_center y_center width height`，坐标归一化到 [0, 1]
- **VOC XML**：管道会自动解析并转换为 YOLO TXT 格式

### 类别文件

在 `data/raw/classes.txt` 中定义类别名称，每行一个：

```
person
car
bicycle
dog
```

如不提供，管道会自动从标注文件中提取类别 ID 并生成默认名称 `class0`、`class1` ...

---

## 使用方法

```bash
# 激活环境后
# 使用 YOLOv5 v5.0（默认）
python yolo_train_pipeline.py

# 指定 Ultralytics YOLOv8（无子版本，直接写模型名）
python yolo_train_pipeline.py --yolo-version yolov8

# 指定 YOLO11
python yolo_train_pipeline.py --yolo-version yolo11
```

管道将依次执行：

1. **初始化配置** — 根据 `--yolo-version` 自动选择 YOLOv5Config 或 UltralyticsConfig
2. **创建目录** — 确保所有输出目录存在
3. **准备数据** — 校验数据 → 格式转换（如需）→ 生成类别文件 → 划分数据集 → 生成 data.yaml
4. **获取权重** — YOLOv5 体系自动下载权重；Ultralytics 体系由包在运行时自动下载
5. **执行训练** — 按版本策略构建命令行参数 → subprocess 启动训练 → 等待完成

训练完成后，结果保存在 `project/train_results/`。

> 数据准备（data.py）在两个体系间完全共享，共用一个 data.yaml。

---

## 架构说明

### 策略模式（Strategy Pattern）

不同 YOLO 版本的训练参数传递方式不同，通过策略模式消除 if/else：

```
get_strategy(identifier)
    │ YOLOv5 体系：
    ├── "5.0" → V5Strategy        # hyp.yaml + --multi-scale --rect 等标志
    ├── "6.0" → V6PlusStrategy     # 全命令行参数 --patience --optimizer ...
    └── "7.0" → V6PlusStrategy     # 与 v6.0 共用
    │ Ultralytics 体系：
    ├── "yolov8" → UltralyticsStrategy  # yolo CLI key=value 格式
    ├── "yolov10" → UltralyticsStrategy # yolo CLI key=value 格式
    └── "yolo11" → UltralyticsStrategy  # yolo CLI key=value 格式
```

添加新版本只需在 [STRATEGY_MAP](file:///d:/auto_yolov5_train/scripts/strategies/__init__.py) 中注册并实现对应策略类。

### 缓存机制

- **YOLOv5 体系**：代码库采用**双模式**缓存（[VersionManager](file:///d:/auto_yolov5_train/scripts/version_manager.py)）：本地优先（`yolov5/v7.0/`），缓存兜底（`~/.cache/yolov5/`）
- **Ultralytics 体系**：无需下载源码，`pip install ultralytics` 即可，权重由包在首次使用 `YOLO(model)` 时自动下载

### 配置冻结

`YOLOv5Config` 和 `UltralyticsConfig` 实例化后自动冻结，内部 dict 替换为 `MappingProxyType`，运行时修改会抛出 `AttributeError`，从机制上防止误修改。

---

## 版本支持对照

### YOLOv5 体系

| 功能 | v5.0 | v6.0 | v7.0 |
|------|------|------|------|
| 数据准备 | ✅ | ✅ | ✅ |
| 自动下载代码 | ✅ | ✅ | ✅ |
| 预训练权重 | ✅ | ✅ | ✅ |
| 超参传递 | hyp.yaml | 命令行参数 | 命令行参数 |
| --multi-scale | ✅ | ❌ | ❌ |
| --rect | ✅ | ❌ | ❌ |
| --label-smoothing | ✅ | ✅ | ✅ |
| --cache ram | ❌ | ✅ | ✅ |
| --patience 早停 | ❌ | ✅ | ✅ |

### Ultralytics 体系

| 功能 | yolov8 | yolov10 | yolo11 |
|------|--------|---------|--------|
| 数据准备 | ✅ | ✅ | ✅ |
| 安装方式 | `pip install ultralytics` | ← 同 | ← 同 |
| 预训练权重 | 自动下载 | 自动下载 | 自动下载 |
| 超参传递 | CLI key=value | CLI key=value | CLI key=value |
| AMP | ✅ | ✅ | ✅ |
| 自动增强 | ✅ | ✅ | ✅ |
| Mosaic | ✅ | ✅ | ✅ |

---

## 常见问题

### Q: 没有 GPU 能训练吗？

可以。将配置中的 `device` 改为 `"cpu"`，但训练速度会非常慢，建议至少 4GB 显存的 NVIDIA GPU。

### Q: 如何选择模型大小？

管道会根据 GPU 显存自动推荐（以 Ultralytics 体系为例，YOLOv5 体系类似）：

| 显存 | 推荐模型 | 说明 |
|------|---------|------|
| < 4GB | `yolov8n.pt` | 最快，精度较低 |
| 4~6GB | `yolov8s.pt` | 小模型 |
| 6~8GB | `yolov8m.pt` | 平衡速度和精度 |
| 8~12GB | `yolov8l.pt` | 高精度，较慢 |
| >= 12GB | `yolov8x.pt` | 最高精度，最慢 |

### Q: `yolov5-5.0`、`yolov8`、`yolo11` 之间有什么区别？

`yolov5-5.0` 中的 `-5.0` 是 **YOLOv5 的子版本号**（YOLOv5 有 v5.0→v6.0→v7.0 三次架构迭代）。而 `yolov8`、`yolov10`、`yolo11` 各只有一个版本，没有子版本号，直接以模型名作为标识即可。
为向后兼容，纯数字写法（`5.0`、`8.0` 等）或带冗余后缀的写法（`yolov8-8.0`）仍可识别，但建议使用无冗余的标准标识。

### Q: YOLOv5 体系和 Ultralytics 体系有什么区别？

| 对比项 | YOLOv5 体系 | Ultralytics 体系 |
|--------|------------|----------------|
| 子版本 | v5.0 / v6.0 / v7.0（有真实子版本） | 无子版本，模型名即标识 |
| 安装 | git clone YOLOv5 源码 | `pip install ultralytics` |
| 执行 | `python train.py` | `yolo train key=value` |
| 配置类 | `YOLOv5Config`（接受 `version` 参数） | `UltralyticsConfig`（接受 `model` 参数） |
| 标识字段 | `YOLO_VERSION` = 版本号（如 `"5.0"`） | `MODEL` = 模型名（如 `"yolov8"`） |
| 权重管理 | 手动下载 | 自动下载 |
| 配置命名 | `batch_size`, `img_size` | `batch`, `imgsz` |

### Q: 训练中途中断了怎么办？

`Ctrl+C` 会安全终止训练（SIGINT 信号处理器负责清理子进程）。重新运行会从上次保存的 checkpoint 恢复（`project/train_results/weights/last.pt`）。

### Q: 网络不好，YOLOv5 代码下载失败？

手动将 YOLOv5 代码克隆到对应目录：

```bash
git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git yolov5/v7.0
```

---

## 许可证

本项目仅供学习研究使用。YOLOv5 代码库遵循 [AGPL-3.0 许可证](https://github.com/ultralytics/yolov5/blob/master/LICENSE)。
