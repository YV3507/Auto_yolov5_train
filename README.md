# Auto YOLOv5 Train Pipeline

YOLOv5 自动化训练管道 —— 一键完成数据准备、版本管理、权重下载和模型训练，支持 YOLOv5 v5.0 / v6.0 / v7.0 全版本。

---

## 功能特性

- **全自动流程** — 数据准备 → 版本代码同步 → 权重下载 → 模型训练，一行命令跑完
- **多版本支持** — 通过策略模式统一适配 YOLOv5 v5.0、v6.0、v7.0，训练参数自动按版本拼装
- **智能缓存** — YOLOv5 代码库通过 git shallow clone 自动缓存，本地优先、缓存兜底
- **权重管理** — 自动下载对应版本的预训练权重，支持断点续传（带进度条）
- **数据预处理** — 支持 YOLO TXT 和 VOC XML 两种标注格式，自动划分训练/验证/测试集
- **配置冻结** — 运行时配置不可变，杜绝意外修改导致的隐式 bug
- **优雅退出** — 注册信号处理器，`Ctrl+C` 安全终止训练
- **双环境支持** — 同时提供 Conda 和 uv 两种环境管理方案

---

## 项目结构

```
auto_yolov5_train/
├── yolov5_train_pipeline.py    # 主入口脚本
├── pyproject.toml              # uv 项目配置与依赖声明
├── .python-version             # Python 版本锁定（3.10）
├── uv.lock                     # uv 依赖锁定文件（可复现安装）
│
├── scripts/
│   ├── config.py               # YOLOv5Config 配置类（版本、路径、超参）
│   ├── data.py                 # 数据准备（划分数据集、格式转换、生成 data.yaml）
│   ├── trainer.py              # 训练执行（subprocess + 信号处理）
│   ├── version_manager.py      # YOLOv5 版本代码缓存 + 预训练权重下载
│   └── strategies/
│       ├── __init__.py         # 策略注册（STRATEGY_MAP）
│       ├── base.py             # VersionStrategy 抽象基类
│       ├── v5.py               # v5.0 策略（hyp.yaml + 命令行标志）
│       └── v6plus.py           # v6.0+/v7.0 策略（全命令行参数）
│
├── data/                       # 数据目录（克隆后自动存在）
│   └── .gitkeep                # 占位文件，使 Git 跟踪空目录
│   └── raw/                    # 用户原始数据，不提交版本库
│       ├── images/             # 图片文件（.jpg / .png）
│       └── labels/             # 标注文件（YOLO .txt 或 VOC .xml）
│
├── project/                    # [运行中生成]训练输出
│   ├── train.txt / val.txt     # 数据集划分文件
│   ├── data.yaml               # YOLO 数据配置
│   ├── hyp.custom.yaml         # 自定义超参文件（v5.0 生成）
│   └── train_results/          # 训练结果（权重、日志、图表）
│
├── weights/                    # 预训练权重缓存
│
├── training.log                # 管道运行日志
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

# 安装训练依赖（如需在本机执行训练）
uv sync --group train

# 激活环境
.venv\Scripts\activate
```

> **关于 PyTorch CUDA**：默认 `uv sync --group train` 从 PyPI 安装的 torch 为 CPU 版本。如需 CUDA 支持，手动安装对应版本：
> ```bash
> # 以 CUDA 12.1 为例
> uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
> ```

### 方式二：Conda

```bash
# 创建环境
conda create -n auto_yolov5_train python=3.10 -y
conda activate auto_yolov5_train

# 安装管道基础依赖
pip install requests tqdm

# 安装 YOLOv5 训练依赖（按需）
pip install torch torchvision opencv-python matplotlib numpy Pillow PyYAML scipy thop tensorboard pandas seaborn psutil gitpython ipython
```

---

## 配置

所有训练参数集中在 [scripts/config.py](file:///d:/auto_yolov5_train/scripts/config.py) 的 `YOLOv5Config` 类中，实例化后自动冻结，运行时不可修改。

### 关键配置项

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
| 数据 | `train_ratio` | `0.8` | 训练集比例 |
| 数据 | `val_ratio` | `0.2` | 验证集比例 |
| 数据 | `annotation_format` | `"txt"` | `"txt"` YOLO 格式 / `"xml"` VOC 格式 |

在 `main()` 中实例化时传入参数即可修改：

```python
cfg = YOLOv5Config(version="7.0")
# 或指定模型
cfg = YOLOv5Config(version="6.0", model="yolov5m.yaml")
```

如需持久化修改，直接编辑 [scripts/config.py](file:///d:/auto_yolov5_train/scripts/config.py) 中的默认值。

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
python yolov5_train_pipeline.py
```

管道将依次执行：

1. **初始化配置** — 加载 YOLOv5Config，检查版本兼容性
2. **创建目录** — 确保所有输出目录存在
3. **准备数据** — 校验数据 → 格式转换（如需）→ 生成类别文件 → 划分数据集 → 生成 data.yaml
4. **获取权重** — 自动推荐模型类型 → 下载对应版本预训练权重
5. **同步代码** — 检查本地 YOLOv5 代码库 → 不存在则自动 git clone
6. **执行训练** — 按版本策略构建命令行参数 → subprocess 启动训练 → 等待完成

训练完成后，结果保存在 `project/train_results/`。

---

## 架构说明

### 策略模式（Strategy Pattern）

不同 YOLOv5 版本的训练参数传递方式不同，通过策略模式消除 if/else：

```
get_strategy(version)
    ├── "5.0" → V5Strategy      # hyp.yaml + --multi-scale --rect 等标志
    └── "6.0" → V6PlusStrategy   # 全命令行参数 --patience --optimizer ...
    └── "7.0" → V6PlusStrategy   # 与 v6.0 共用
```

添加新版本只需在 [STRATEGY_MAP](file:///d:/auto_yolov5_train/scripts/strategies/__init__.py) 中注册并实现对应策略类。

### 缓存机制

YOLOv5 代码库采用**双模式**缓存（[VersionManager](file:///d:/auto_yolov5_train/scripts/version_manager.py)）：

- **本地优先**：`yolov5/v7.0/` 等目录存在时直接使用
- **缓存兜底**：自动 git shallow clone 到 `~/.cache/yolov5/v{version}/`

### 配置冻结

`YOLOv5Config` 实例化后自动冻结，内部 dict 替换为 `MappingProxyType`，运行时修改会抛出 `AttributeError`，从机制上防止误修改。

---

## 版本支持对照

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

---

## 常见问题

### Q: 没有 GPU 能训练吗？

可以。将配置中的 `device` 改为 `"cpu"`，但训练速度会非常慢，建议至少 4GB 显存的 NVIDIA GPU。

### Q: 如何选择模型大小？

管道会根据 GPU 显存自动推荐：

| 显存 | 推荐模型 | 说明 |
|------|---------|------|
| < 4GB | `yolov5s.pt` | 最快，精度较低 |
| 4~8GB | `yolov5m.pt` | 平衡速度和精度 |
| 8~12GB | `yolov5l.pt` | 高精度，较慢 |
| >= 12GB | `yolov5x.pt` | 最高精度，最慢 |

### Q: 训练中途中断了怎么办？

`Ctrl+C` 会安全终止训练（SIGINT 信号处理器负责清理子进程）。重新运行会从上次保存的 checkpoint 恢复（`project/train_results/weights/last.pt`）。

### Q: 网络不好，YOLOv5 代码下载失败？

方案一：手动将 YOLOv5 代码克隆到对应目录：

```bash
git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git yolov5/v7.0
```

---

## 许可证

本项目仅供学习研究使用。YOLOv5 代码库遵循 [AGPL-3.0 许可证](https://github.com/ultralytics/yolov5/blob/master/LICENSE)。
