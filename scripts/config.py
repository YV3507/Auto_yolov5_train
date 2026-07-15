"""
YOLOv5 / Ultralytics 训练配置管理模块

双体系并立，互不干扰：
  - YOLOv5 体系：YOLOv5Config(version="5.0") — 有子版本，接受版本号
  - Ultralytics 体系：UltralyticsConfig(model="yolov8") — 无子版本，接受模型名

共享基类 BaseConfig 提供：
  - 冻结保护机制（FreezeableConfigMixin）
  - 公共路径配置
  - 公共数据划分配置

使用方法：
    from scripts.config import YOLOv5Config, UltralyticsConfig

    # YOLOv5
    cfg = YOLOv5Config(version="5.0")
    cfg = YOLOv5Config(version="7.0", model="yolov5m.yaml")

    # Ultralytics
    cfg = UltralyticsConfig(model="yolov8")
    cfg = UltralyticsConfig(model="yolo11")

注意事项：
    - 两配置类实例化后均自动冻结，运行时修改会抛出 AttributeError
    - YOLOv5Config.YOLO_VERSION 可选值: "5.0", "6.0", "7.0"
    - UltralyticsConfig.MODEL 可选值: "yolov8", "yolov10", "yolo11"
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Optional


class BaseConfig:
    """配置基类：提供冻结保护 + 公共路径 + 公共数据配置"""

    def __init__(self):
        """初始化基类配置：设置项目根目录、公共路径和公共数据划分配置。"""
        self._frozen = False

        # ==================== 项目根目录 ====================

        # 自动检测项目根目录（config.py 位于 scripts/ 下，其父目录的父目录即为项目根）
        root = Path(__file__).parent.parent
        self.PROJECT_ROOT = root

        # ==================== 公共路径配置 ====================

        self.PATHS = {
            # 原始数据根目录 —— 存放未处理前的原始训练数据
            'raw_data': root / "data" / "raw",
            # 原始图片目录 —— 所有训练/验证图片（.jpg/.jpeg/.png/.bmp），建议统一放入此目录
            'images_dir': root / "data" / "raw" / "images",
            # 原始标注目录 —— 存放 YOLO TXT 或 VOC XML 标注文件，文件名与图片名对应（不含扩展名）
            'labels_dir': root / "data" / "raw" / "labels",
            # 类别名称文件 —— 每行一个类别名，如 turn_left / crossing / limit_10 ...
            # 文件路径固定，自动生成或由用户提供。行号即类别 ID（从 0 开始）。
            'classes_file': root / "data" / "raw" / "classes.txt",
            # 训练输出根目录 —— 输出 data.yaml / hyp.custom.yaml / train.txt / val.txt
            # 以及训练结果目录 train_results/（含模型权重、训练曲线等）
            'project_dir': root / "project",
            # 预训练权重存放目录 —— 如 yolov5s.pt / yolov5l.pt，自动下载到此目录
            'weights_dir': root / "weights",
        }

        # ==================== 公共数据配置 ====================

        self.DATA = {
            # 标注文件格式：'txt'（YOLO 格式，推荐）或 'xml'（VOC 格式，自动转换为 YOLO 格式）
            'annotation_format': 'txt',
            # 训练集划分比例（默认 0.8 = 80% 数据用于训练）
            'train_ratio': 0.8,
            # 验证集划分比例（默认 0.2 = 20% 数据用于验证）
            'val_ratio': 0.2,
            # 测试集划分比例（默认 0.0 = 不划分测试集，所有数据合并为训练+验证）
            # 如需测试集，建议设为 0.1（此时需同步调整 train_ratio 和 val_ratio，使三者之和为 1.0）
            'test_ratio': 0.0,
            # 随机种子 —— 确保数据集划分结果可复现。修改此值可获得不同的随机划分结果。
            'random_seed': 42,
        }

    # ==================== 冻结保护（子类初始化完成后调用 _freeze()）====================

    def _freeze(self):
        """
        递归冻结所有内部 dict，禁止运行时修改。

        将 self.__dict__ 中所有 dict 类型的属性替换为 MappingProxyType（只读代理），
        之后任何尝试修改字典内容的操作都会抛出 TypeError。
        同时设置 _frozen = True，使 __setattr__ 拦截新增/修改非私有属性的操作。
        """
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, types.MappingProxyType(value))
        self._frozen = True

    def __setattr__(self, name, value):
        """
        冻结保护：实例冻结后禁止添加或修改非私有属性。

        允许的操作：
          - 设置 _frozen / _freezed 等私有属性（以下划线开头）
          - 冻结之前的任何属性设置（由子类 __init__ 完成）

        禁止的操作：
          - 冻结后尝试新增属性
          - 冻结后尝试修改已有属性（AttributeError 异常）
        """
        if getattr(self, '_frozen', False) and not name.startswith('_'):
            raise AttributeError(f"配置已冻结，不允许运行时修改属性: {name}")
        super().__setattr__(name, value)


class YOLOv5Config(BaseConfig):
    """
    YOLOv5 训练配置类，实例化后冻结。

    支持的 YOLOv5 版本：5.0 / 6.0 / 7.0，不同版本的训练参数传递方式不同：
      - v5.0：通过 hyp.yaml 文件传递超参，通过命令行标志控制功能开关
      - v6.0+：所有超参通过命令行参数直接传递
    """

    # 支持的 YOLOv5 版本号白名单
    ALLOWED_VERSIONS = {"5.0", "6.0", "7.0"}

    def __init__(self, version: str = "5.0", model: Optional[str] = None):
        """
        初始化 YOLOv5 训练配置。

        Args:
            version: YOLOv5 版本号，可选值: "5.0", "6.0", "7.0"
            model:   模型配置文件名称，如 "yolov5s.yaml" / "yolov5m.yaml" / "yolov5l.yaml" / "yolov5x.yaml"
                    为 None 时默认使用 "yolov5s.yaml"（最小模型，适合快速验证）
        """
        super().__init__()  # 设置 _frozen=False, PROJECT_ROOT, PATHS, DATA

        # ==================== 版本验证 ====================

        if version not in self.ALLOWED_VERSIONS:
            raise ValueError(
                f"不支持的 YOLOv5 版本: {version}，可选: {sorted(self.ALLOWED_VERSIONS)}"
            )
        # YOLOv5 版本标识，用于适配不同版本的训练参数传递方式和权重下载 URL
        self.YOLO_VERSION = version

        root = self.PROJECT_ROOT

        # ==================== 扩展路径（仅 YOLOv5 需要）====================

        # WandB 日志输出目录（用于训练过程可视化，需安装 wandb 包）
        self.PATHS['wandb_dir'] = root
        # YOLOv5 各版本代码库路径 —— 由 VersionManager 统一管理下载/缓存
        # 路径存在则直接使用；不存在时自动 git clone --depth 1 下载到此处
        self.PATHS['yolov5_5_0'] = root / "yolov5" / "v5.0"
        self.PATHS['yolov5_6_0'] = root / "yolov5" / "v6.0"
        self.PATHS['yolov5_7_0'] = root / "yolov5" / "v7.0"

        # ==================== 训练超参数配置 ====================

        self.TRAINING = {
            # 模型配置文件名称（位于 YOLOv5 源码的 models/ 目录下）
            #   yolov5s.yaml — 最小模型（~7.2M 参数），适合快速验证和低显存 GPU
            #   yolov5m.yaml — 中等模型（~21.2M 参数），平衡速度与精度
            #   yolov5l.yaml — 大模型（~46.5M 参数），高精度
            #   yolov5x.yaml — 最大模型（~86.7M 参数），最高精度（需大显存）
            'model': model or 'yolov5s.yaml',

            # 批大小（Batch Size） —— 每轮迭代处理的样本数
            # 根据 GPU 显存调整建议：
            #   RTX 4060 8GB → 16~24
            #   RTX 3090 24GB → 32~64
            #   显存不足时降低此值，但建议不低于 8 以保证 BN 层统计稳定
            'batch': 16,

            # 训练总轮数（Epochs）
            # 建议：
            #   小数据集（<1000 张）→ 300~500 epochs（配合早停）
            #   中等数据集 → 120~200 epochs
            #   大数据集（>10000 张）→ 80~120 epochs
            'epochs': 120,

            # 输入图像尺寸（像素，正方形）
            #   640 — 标准 YOLOv5 输入尺寸，适合大多数场景
            #   320 — 更快但精度下降，适合移动端/实时场景
            #   1280 — 更高精度但显存消耗大幅增加，适合小目标检测
            'imgsz': 640,

            # 训练设备
            #   '0'      — 使用第一块 GPU（推荐）
            #   'cpu'    — 使用 CPU 训练（极慢，仅适合模型验证）
            #   '0,1'    — 使用多 GPU 数据并行（实验性，稳定性较差）
            'device': '0',

            # 数据加载进程数（DataLoader num_workers）
            # 建议设置为 CPU 核心数或略低：
            #   16 核 CPU + 32GB 内存 → 12
            #   8 核 CPU → 6~8
            #   内存不足时适当降低
            'workers': 12,

            # 早停耐心值（仅 v6.0+ 有效，v5.0 会自动忽略）
            #   含义：验证集 mAP 连续 N 个 epoch 未提升时停止训练
            #   设为 0 禁用早停
            'patience': 10,

            # 【v5.0 支持】多尺度训练（Multi-Scale Training）
            #   True  — 每个 batch 的图像尺寸在 [0.5×imgsz, 1.5×imgsz] 之间随机变化
            #           可提升模型对不同尺度目标的鲁棒性，充分利用 GPU 算力
            #   False — 所有图像固定为 imgsz 尺寸
            'multi_scale': True,

            # 【v5.0 支持】矩形训练（Rectangular Training）
            #   True  — 根据图像原始宽高比进行填充，减少无效填充区域
            #           可提高训练速度约 10~20%，对精度无显著影响
            #   False — 所有图像强制缩放为正方形
            'rect': True,

            # 【v5.0 支持】标签平滑系数（Label Smoothing，范围 0.0~0.1）
            #   0.0   — 不使用标签平滑
            #   0.05  — 温和平滑，可提升泛化能力，防止过拟合（推荐）
            #   0.1   — 强平滑，适用于噪声较大的数据集
            'label_smoothing': 0.05,

            # GPU 显存大小（GB）—— 用于 YOLOv5 的 get_recommended_model() 自动推荐模型
            #   < 4GB  → yolov5s.pt（最小模型）
            #   4~8GB  → yolov5m.pt（中等模型）
            #   8~12GB → yolov5l.pt（大模型）
            #   ≥ 12GB → yolov5x.pt（最大模型）
            'gpu_memory': 8,
        }

        # ==================== 优化器配置 ====================

        self.OPTIMIZER = {
            # 优化器类型（v6.0+ 完整支持；v5.0 通过 --adam 标志控制）
            #   'SGD'   — 随机梯度下降（默认，收敛效果好，泛化能力强）
            #   'Adam'  — Adam 优化器（收敛快，但可能需要调低学习率）
            #   'AdamW' — Adam with Decoupled Weight Decay（v6.0+，改进版 Adam）
            'optimizer': 'SGD',

            # 【v5.0 支持】是否使用 Adam 优化器（True → 添加 --adam 参数）
            #   True  — 使用 Adam，此时 optimizer 字段被忽略
            #   False — 使用 optimizer 字段指定的优化器
            'use_adam': False,

            # 初始学习率（Initial Learning Rate）
            #   SGD 建议：0.01（默认）
            #   Adam 建议：0.001（需同时调低）
            #   增大 batch_size 时建议按比例提高 lr0（线性缩放规则）
            'lr0': 0.01,

            # 最终学习率系数（Final LR factor），最终学习率 = lr0 × lrf
            #   0.2  — 学习率最终衰减到初始值的 20%（默认，适用于大多数情况）
            #   0.01 — 学习率几乎衰减到 0（需要更长的训练轮数）
            'lrf': 0.2,

            # SGD / AdamW 动量（Momentum）
            #   0.937 — YOLOv5 默认值，经验值
            #   0.9   — 标准动量值，更保守
            'momentum': 0.937,

            # 权重衰减系数（Weight Decay / L2 正则化）
            #   0.0005 — YOLOv5 默认值，防止过拟合
            #   0.001  — 更强正则化，适用于小数据集
            #   0.0    — 无正则化，适用于大数据集
            'weight_decay': 0.0005,

            # 热身轮数（Warmup Epochs）—— 训练初期学习率从很低值逐渐增长到 lr0
            #   3.0 — YOLOv5 默认值
            #   0.0 — 禁用热身（不推荐，可能导致初期训练不稳定）
            'warmup_epochs': 3.0,

            # 热身阶段的初始动量（低于正常动量值）
            #   0.8 — 训练初期动量较低，帮助参数快速逃离初始状态
            'warmup_momentum': 0.8,

            # 热身阶段偏置项的学习率系数（偏置参数使用独立的学习率）
            #   0.1 — 偏置参数学习率 = lr0 × 0.1，避免初期偏置剧烈变化
            'warmup_bias_lr': 0.1,
        }

        # ==================== 数据增强配置 ====================
        # 这些参数会被写入 hyp.custom.yaml 文件（v5.0 通过 --hyp 传递）
        # 或通过命令行参数直接传递（v6.0+）。
        # 详细说明参考：https://github.com/ultralytics/yolov5/wiki/hyperparameters

        self.AUGMENTATION = {
            # -------------------- 色彩空间增强 --------------------
            # 色调（Hue）增强最大偏移量（比例，0~1）
            #   0.015 — 轻微色调扰动，使模型对颜色变化更鲁棒
            #   0.0   — 禁用色调增强
            'hsv_h': 0.015,

            # 饱和度（Saturation）增强最大偏移量（比例，0~1）
            #   0.7   — 较强饱和度扰动，增强对不同光照条件的适应性
            #   0.0   — 禁用饱和度增强
            'hsv_s': 0.7,

            # 明度（Value）增强最大偏移量（比例，0~1）
            #   0.4   — 中等明度扰动，模拟不同曝光条件
            #   0.0   — 禁用明度增强
            'hsv_v': 0.4,

            # -------------------- 几何变换增强 --------------------
            # 随机旋转角度（度）
            #   0.0   — 不旋转（适用于方向固定的场景如交通标志、车牌）
            #   10.0  — 轻微随机旋转
            #   45.0  — 大幅度随机旋转（适用于通用目标检测）
            'degrees': 0.0,

            # 随机平移比例（相对于图像尺寸）
            #   0.1   — 允许图像在水平和垂直方向最多平移 10%
            #   0.0   — 禁用平移
            # 注意：过大的平移可能导致目标移出图像边界
            'translate': 0.1,

            # 随机缩放比例（因子范围 1±scale）
            #   0.5   — 缩放比例在 [0.5, 1.5] 之间随机
            #   0.0   — 禁用缩放增强
            'scale': 0.5,

            # 随机剪切强度（度）
            #   0.0   — 不剪切（推荐，剪切对目标检测帮助有限）
            'shear': 0.0,

            # 随机透视变换强度（因子）
            #   0.0   — 禁用（推荐，透视变换会显著降低训练性能）
            'perspective': 0.0,

            # -------------------- 翻转增强 --------------------
            # 垂直翻转概率（0~1）
            #   0.0   — 不垂直翻转（交通标志方向固定，不适合垂直翻转）
            #   0.5   — 50% 概率垂直翻转（适用于航拍/通用场景）
            'flipud': 0.0,

            # 水平翻转概率（0~1）
            #   0.0   — 不水平翻转（交通标志中文字方向固定）
            #   0.5   — 50% 概率水平翻转（推荐，对绝大多数目标检测有效）
            'fliplr': 0.0,

            # -------------------- 高级数据增强 --------------------
            # Mosaic 数据增强概率 —— 将 4 张训练图拼接为 1 张马赛克图
            #   1.0   — 100% 概率启用（YOLOv5 默认，可大幅提升小目标检测效果）
            #   0.0   — 禁用
            # 注意：mosaic 会增加显存占用，训练后期可适当降低或配合 close_mosaic
            'mosaic': 1.0,

            # MixUp 数据增强概率（将 2 张图按比例混合）
            #   0.0   — 禁用（v5.0 支持有限，建议保持 0）
            #   0.1~0.2 — 轻微 MixUp，可提升泛化能力（v6.0+ 支持）
            'mixup': 0.0,
        }

        # ==================== 预训练权重配置 ====================

        self.WEIGHTS = {
            # 用户手动指定的预训练模型文件名（优先级高于自动推荐）
            #   例如 'yolov5l.pt' / 'yolov5s.pt' 等
            #   设为 None 或空字符串时，由 get_recommended_model() 根据 GPU 显存自动推荐
            'pretrained_model': 'yolov5s.pt',

            # 本地缺少预训练权重时是否自动从 GitHub Releases 下载
            #   True  — 自动下载（推荐，使用预训练权重可显著提升收敛速度）
            #   False — 跳过下载，从头开始训练（需大量数据和更长训练时间）
            'download_missing': True,

            # 本地缺少 YOLOv5 代码库时是否自动从 GitHub 下载
            #   True  — 自动 git clone（推荐）
            #   False — 需手动准备代码库
            'auto_download_yolov5': True,
        }

        # ==================== 日志配置 ====================

        self.LOGGING = {
            # 日志级别
            #   'DEBUG'   — 输出最详细日志（含参数调试信息）
            #   'INFO'    — 标准日志输出（推荐）
            #   'WARNING' — 仅输出警告和错误
            #   'ERROR'   — 仅输出错误信息
            'log_level': 'INFO',

            # 日志文件保存路径（自动创建父目录）
            'log_file': root / "training.log",
        }

        # ---- 冻结配置，禁止运行时修改 ----
        self._freeze()

    # ==================== 工具方法 ====================

    def get_version_key(self) -> str:
        """
        版本号转路径键。

        将配置中的版本号（如 '5.0'）转换为 PATHS 字典中对应的键名（如 'yolov5_5_0'）。
        此方法集中化版本→路径的映射逻辑，避免各模块重复编写转换代码。

        Returns:
            路径键字符串，如 'yolov5_5_0'
        """
        return f"yolov5_{self.YOLO_VERSION.replace('.', '_')}"

    def get_yolov5_path(self) -> Path:
        """
        获取当前版本对应的 YOLOv5 代码库路径。

        返回的路径将作为 VersionManager.ensure() 的 local_dir 参数：
          - 目录存在 → 直接使用
          - 目录不存在 → 由 VersionManager git clone 到此路径

        Returns:
            代码库目录的 Path 对象，如 D:/auto_yolov5_train/yolov5/v5.0
        """
        return self.PATHS[self.get_version_key()]

    def get_yolov5_paths(self) -> list[tuple[str, Path]]:
        """
        列出项目目录下所有已存在的 YOLOv5 版本。

        用于日志输出，提示用户当前可用的 YOLOv5 版本。

        Returns:
            列表，每个元素为 (版本号字符串, Path对象) 的元组
            如 [('5.0', Path('.../yolov5/v5.0')), ('6.0', Path('.../yolov5/v6.0'))]
        """
        result = []
        for key, path in self.PATHS.items():
            if key.startswith('yolov5_') and path.exists():
                version = '.'.join(key.split('_')[1:])
                result.append((version, path))
        return result

    def get_recommended_model(self, gpu_memory_gb: Optional[float] = None) -> str:
        """
        获取预训练模型文件名（用户指定优先，否则根据显存自动推荐）。

        优先级：
          1. WEIGHTS['pretrained_model'] 非空 → 直接返回用户指定值
          2. 否则根据 GPU 显存自动推荐

        显存区间与自动推荐的对应关系：
            < 4GB   → yolov5s.pt
            4~8GB   → yolov5m.pt
            8~12GB  → yolov5l.pt（适合 RTX 4060 等 8GB 主流卡）
            ≥ 12GB  → yolov5x.pt（适合 RTX 3090/4090）

        Args:
            gpu_memory_gb: GPU 显存大小（GB）。为 None 时自动使用 TRAINING['gpu_memory']。

        Returns:
            模型文件名，如 'yolov5l.pt'
        """
        manual = self.WEIGHTS.get('pretrained_model')
        if manual:
            return manual
        mem = gpu_memory_gb if gpu_memory_gb is not None else self.TRAINING.get('gpu_memory', 8)
        if mem < 4:
            return 'yolov5s.pt'
        elif mem < 8:
            return 'yolov5m.pt'
        elif mem < 12:
            return 'yolov5l.pt'
        else:
            return 'yolov5x.pt'


class UltralyticsConfig(BaseConfig):
    """
    Ultralytics 训练配置类（YOLOv8 / YOLOv10 / YOLO11）。

    YOLOv5 有真实子版本（v5.0 → v6.0 → v7.0），因此 YOLOv5Config 接受 version 参数。
    Ultralytics 各模型无子版本，因此本类接受 model 参数（模型名即标识）。

    支持的模型：
      - yolov8  — YOLOv8（成熟稳定，生态完善）
      - yolov10 — YOLOv10（最新架构，精度更高）
      - yolo11  — YOLO11（Ultralytics 最新系列）
    """

    # 支持的模型名称白名单
    ALLOWED_MODELS = {"yolov8", "yolov10", "yolo11"}

    # 模型大小后缀（由小到大，参数量和精度递增）
    MODEL_SIZES = ("n", "s", "m", "l", "x")

    def __init__(self, model: str = "yolov8"):
        """
        初始化 Ultralytics 训练配置。

        Args:
            model: 模型名称，可选值: "yolov8", "yolov10", "yolo11"
        """
        super().__init__()  # 设置 _frozen=False, PROJECT_ROOT, PATHS, DATA

        # ==================== 模型验证 ====================

        if model not in self.ALLOWED_MODELS:
            raise ValueError(
                f"不支持的 Ultralytics 模型: {model}，可选: {sorted(self.ALLOWED_MODELS)}"
            )
        # Ultralytics 模型标识（如 "yolov8" / "yolo11"）
        self.MODEL = model
        # 与 YOLOv5Config 接口对齐，但此处语义为"模型标识"而非版本号
        self.YOLO_VERSION = model

        # ==================== 训练超参数配置（Ultralytics 原生参数名）====================

        self.TRAINING = {
            # 模型权重文件名（自动根据模型名和默认大小 'n' 生成）
            #   例如 yolov8 → "yolov8n.pt"，yolo11 → "yolo11n.pt"
            #   模型大小可选: n（nano, 最小） / s（small） / m（medium） / l（large） / x（xlarge）
            'model': f'{model}n.pt',

            # 输入图像尺寸（像素，正方形）
            #   640 — 标准尺寸，YOLOv8 默认
            'imgsz': 640,

            # 批大小（Batch Size）
            #   16 — YOLOv8 默认值，8GB 显存可稳定运行
            'batch': 16,

            # 训练总轮数
            'epochs': 120,

            # 训练设备：'0'（GPU）或 'cpu'
            'device': '0',

            # 数据加载进程数
            'workers': 12,

            # 早停耐心值（验证集指标连续 N 轮未提升则停止）
            'patience': 10,

            # 优化器类型
            #   'SGD'     — 随机梯度下降（默认，泛化能力强）
            #   'Adam'    — Adam 优化器（收敛快）
            #   'AdamW'   — Adam with 解耦权重衰减
            #   'auto'    — 自动选择（根据模型自动判断）
            'optimizer': 'SGD',

            # 初始学习率
            'lr0': 0.01,

            # 最终学习率系数（lr0 × lrf）
            'lrf': 0.2,

            # 动量
            'momentum': 0.937,

            # 权重衰减系数
            'weight_decay': 0.0005,

            # 热身轮数
            'warmup_epochs': 3.0,

            # 热身阶段动量
            'warmup_momentum': 0.8,

            # 热身阶段偏置学习率系数
            'warmup_bias_lr': 0.1,

            # 是否使用余弦退火学习率调度
            #   False — 使用线性衰减（默认）
            #   True  — 使用余弦退火调度
            'cos_lr': False,

            # Mosaic 增强关闭轮数（最后 N 轮关闭 mosaic，提升精度）
            #   10 — 最后 10 个 epoch 关闭 mosaic 增强
            'close_mosaic': 10,

            # 训练过程中是否在验证集上评估
            'val': True,

            # 是否使用混合精度训练（AMP，Automatic Mixed Precision）
            #   True  — 启用（推荐，可减少显存占用约 30~40%，速度提升约 20%）
            #   False — 使用全精度训练
            'amp': True,

            # 使用数据集的子集比例（1.0 = 使用全部数据）
            #   0.1~0.5 — 快速验证训练流程时使用
            #   1.0     — 完整训练
            'fraction': 1.0,

            # 是否保存训练结果（权重文件、日志等）
            'save': True,

            # 数据集缓存模式
            #   False — 不缓存（默认，节省内存）
            #   True  — 缓存到内存（加速训练，但增加内存占用）
            #   'ram' — 同 True
            #   'disk'— 缓存到磁盘
            'cache': False,

            # 是否使用预训练权重
            #   True  — 加载预训练权重（推荐，加快收敛）
            #   False — 从头开始训练（随机初始化）
            'pretrained': True,
        }

        # ==================== 数据增强配置（Ultralytics 原生参数名）====================

        self.AUGMENTATION = {
            # 色调扰动
            'hsv_h': 0.015,
            # 饱和度扰动
            'hsv_s': 0.7,
            # 明度扰动
            'hsv_v': 0.4,

            # 随机旋转（度）
            'degrees': 0.0,
            # 随机平移
            'translate': 0.1,
            # 随机缩放
            'scale': 0.5,
            # 随机剪切
            'shear': 0.0,
            # 透视变换
            'perspective': 0.0,

            # 垂直翻转概率
            'flipud': 0.0,
            # 水平翻转概率（推荐 0.5，交通标志场景保持 0.0）
            'fliplr': 0.5,

            # Mosaic 增强概率
            'mosaic': 1.0,
            # MixUp 增强概率
            'mixup': 0.0,

            # Copy-Paste 增强概率（将一张图中的目标复制粘贴到另一张图）
            #   0.0 — 禁用
            #   0.1~0.3 — 对小目标检测有帮助，但增加训练时间
            'copy_paste': 0.0,

            # 自动数据增强策略
            #   'randaugment' — 随机增强（YOLOv8 默认）
            #   'auto'        — 自动搜索最优增强策略
            #   None          — 禁用自动增强
            'auto_augment': 'randaugment',

            # 随机擦除（Random Erasing）概率
            #   0.4  — 以 40% 概率随机擦除图像中的矩形区域（提升遮挡鲁棒性）
            #   0.0  — 禁用
            'erasing': 0.4,
        }

        # ==================== 权重配置 ====================

        self.WEIGHTS = {
            # 用户手动指定的预训练模型文件名（优先级高于自动推荐）
            #   例如 'yolov8n.pt' / 'yolo11m.pt' 等
            #   设为 None 或空字符串时，由 get_recommended_model() 根据 GPU 显存自动推荐
            'pretrained_model': None,

            # 本地缺少预训练权重时是否自动下载（ultralytics 包自动管理下载）
            'download_missing': True,
        }

        # ---- 冻结 ----
        self._freeze()

    # ==================== 工具方法 ====================

    @classmethod
    def get_model_name(cls, model: str, size: str = 'n') -> str:
        """
        获取 Ultralytics 体系下的完整模型文件名。

        根据模型名和大小后缀拼接完整的 .pt 文件名。
        模型大小与参数量、精度和速度的对应关系：
          n（nano）— 最小，最快，精度最低
          s（small）— 小模型
          m（medium）— 中等，平衡
          l（large）— 大模型，高精度
          x（xlarge）— 最大，最高精度，最慢

        Args:
            model: 模型名，如 "yolov8", "yolo11"
            size: 模型大小后缀，可选: "n", "s", "m", "l", "x"

        Returns:
            模型文件名，如 "yolov8n.pt"

        Raises:
            ValueError: 不支持的模型名或模型大小
        """
        if model not in cls.ALLOWED_MODELS:
            raise ValueError(f"不支持的模型: {model}，可选: {sorted(cls.ALLOWED_MODELS)}")
        if size not in cls.MODEL_SIZES:
            raise ValueError(f"不支持的模型大小: {size}，可选: {cls.MODEL_SIZES}")
        return f"{model}{size}.pt"

    def get_recommended_model(self, gpu_memory_gb: Optional[float] = None) -> str:
        """
        获取预训练模型文件名（用户指定优先，否则根据显存自动推荐）。

        优先级：
          1. WEIGHTS['pretrained_model'] 非空 → 直接返回用户指定值
          2. 否则根据 GPU 显存自动推荐

        显存区间与自动推荐的对应关系：
            < 4GB   → nano（如 yolov8n.pt）
            4~6GB   → small（如 yolov8s.pt）
            6~8GB   → medium（如 yolov8m.pt）
            8~12GB  → large（如 yolov8l.pt）
            ≥ 12GB  → xlarge（如 yolov8x.pt）

        Args:
            gpu_memory_gb: GPU 显存（GB），为 None 时使用默认值 8

        Returns:
            模型文件名，如 "yolov8n.pt"
        """
        manual = self.WEIGHTS.get('pretrained_model')
        if manual:
            return manual
        mem = gpu_memory_gb if gpu_memory_gb is not None else 8
        size = (
            'n' if mem < 4 else
            's' if mem < 6 else
            'm' if mem < 8 else
            'l' if mem < 12 else
            'x'
        )
        return self.get_model_name(self.MODEL, size)

    def build_cli_args(self) -> list[str]:
        """
        将配置序列化为 Ultralytics CLI key=value 格式的参数列表。

        生成的参数可直接传递给 `yolo train key=value` 命令。
        参数包括：基础训练参数、布尔开关参数、数据增强参数、数据路径和输出路径。

        Returns:
            CLI 参数列表，如 ['imgsz=640', 'batch=16', 'epochs=120', ...]
        """
        args = []

        # 基础训练参数（标量值）
        for key in ['imgsz', 'batch', 'epochs', 'device', 'workers',
                     'patience', 'optimizer', 'lr0', 'lrf', 'momentum',
                     'weight_decay', 'warmup_epochs', 'warmup_momentum',
                     'warmup_bias_lr', 'close_mosaic', 'fraction', 'cache']:
            args.append(f'{key}={self.TRAINING[key]}')

        # 布尔值参数（需转换为小写字符串 'true'/'false'）
        for key in ['cos_lr', 'val', 'amp', 'save', 'pretrained']:
            args.append(f'{key}={str(self.TRAINING[key]).lower()}')

        # 数据增强参数
        for key, value in self.AUGMENTATION.items():
            args.append(f'{key}={value}')

        # 数据配置文件路径
        args.append(f'data={self.PATHS["project_dir"] / "data.yaml"}')
        # 训练输出项目目录
        args.append(f'project={self.PATHS["project_dir"]}')
        # 训练结果子目录名称
        args.append('name=train_results')
        # 允许覆盖已存在的同名目录（不报错，自动覆盖）
        args.append('exist_ok=True')

        return args
