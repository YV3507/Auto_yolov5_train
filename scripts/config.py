"""
YOLOv5 训练配置管理模块

使用方法：
    from scripts.config import YOLOv5Config
    cfg = YOLOv5Config(version="5.0")

注意事项：
    - 实例化后配置自动冻结，运行时修改属性或内部 dict 会抛出异常
    - 如需修改配置，请在实例化时传入参数
    - YOLO_VERSION 可选值: "5.0", "6.0", "7.0"
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Optional


class YOLOv5Config:
    """YOLOv5 训练配置类，实例化后冻结，禁止运行时修改"""

    # 支持的版本号白名单
    ALLOWED_VERSIONS = {"5.0", "6.0", "7.0"}

    def __init__(self, version: str = "5.0", model: Optional[str] = None):
        """
        初始化配置

        Args:
            version: YOLOv5 版本号，用于自动适配不同版本的训练参数
                    可选值: "5.0", "6.0", "7.0"
            model:   模型配置文件名称，如 "yolov5s.yaml"
                    为 None 时使用默认值 "yolov5s.yaml"
        """
        self._frozen = False

        # ==================== 版本验证 ====================
        if version not in self.ALLOWED_VERSIONS:
            raise ValueError(
                f"不支持的 YOLOv5 版本: {version}，可选: {sorted(self.ALLOWED_VERSIONS)}"
            )
        self.YOLO_VERSION = version

        # ==================== 路径配置 ====================
        # 项目根目录（自动检测，无需修改）
        root = Path(__file__).parent.parent
        self.PROJECT_ROOT = root

        # 所有路径均相对于项目根目录
        # 键名中的版本号使用下划线代替点（如 yolov5_5_0）
        # YOLOv5 代码库由 VersionManager 统一管理：
        #   - 指定了 local_dir（即此处的路径）→ 下载到该项目目录下
        #   - 未指定或路径不存在 → 自动下载到 ~/.cache/yolov5/
        self.PATHS = {

            # ----- 原始数据路径（未处理前）-----
            'raw_data': root / "data" / "raw",                    # 原始数据根目录
            'images_dir': root / "data" / "raw" / "images",       # 原始图片目录
            'labels_dir': root / "data" / "raw" / "labels",       # 原始标注目录（YOLO 格式 txt）
            'classes_file': root / "data" / "raw" / "classes.txt",  # 类别名称文件，每行一个类别

            # ----- 输出路径 -----
            'project_dir': root / "project",   # 训练输出根目录（包含 data.yaml, hyp.custom.yaml, train_results 等）
            'weights_dir': root / "weights",   # 预训练权重存放目录（如 yolov5s.pt）
            'wandb_dir': root ,       # wandb 运行日志统一目录，默认为 project_root/wandb/

            # ----- YOLOv5 代码库路径（作为 local_dir 传给 VersionManager）-----
            'yolov5_5_0': root / "yolov5" / "v5.0",  # 版本代码目录
            'yolov5_6_0': root / "yolov5" / "v6.0",
            'yolov5_7_0': root / "yolov5" / "v7.0",
        }

        # ==================== 数据配置 ====================
        self.DATA = {
            'annotation_format': 'txt',    # 标注文件格式：'txt' (YOLO 格式) 或 'xml' (VOC 格式)
            'train_ratio': 0.8,            # 训练集比例（剩余作为验证集）
            'val_ratio': 0.2,              # 验证集比例
            'test_ratio': 0.0,             # 测试集比例（通常设为 0，使用验证集评估即可）
            'random_seed': 42,             # 随机种子，确保数据集划分可复现
        }

        # ==================== 训练配置 ====================
        self.TRAINING = {

            # ----- 模型结构 -----
            'model': model or 'yolov5s.yaml',  # 模型配置文件名称（位于 yolov5/models/ 目录下）
                                                # 可选: yolov5s.yaml (最小), yolov5m.yaml,
                                                #       yolov5l.yaml, yolov5x.yaml (最大)

            # ----- 训练超参数 -----
            'batch_size': 16,               # 批大小（Batch Size）
                                            # 根据 GPU 显存调整：RTX 4060 8GB 建议 16-24
            'epochs': 120,                  # 训练总轮数
            'img_size': 640,                # 输入图像尺寸（像素，正方形）
                                            # 增大可提高精度但增加显存和计算量
            'device': '0',                  # GPU 设备 ID：'0' 表示第一块 GPU，
                                            # 'cpu' 表示 CPU，多卡如 '0,1' 不推荐（稳定性差）
            'workers': 12,                  # 数据加载进程数（DataLoader 的 num_workers）
                                            # CPU 核心充足（16核）且内存充裕（32GB）时可提高

            # ----- 优化参数 -----
            'patience': 10,                 # 早停耐心值（仅 v6.0+ 有效，v5.0 会自动忽略）
            'multi_scale': True,            # 【v5.0 支持】是否开启多尺度训练
                                            # 开启后每个 batch 的图像尺寸在 [0.5*img_size, 1.5*img_size] 随机变化
                                            # 可增加模型鲁棒性，充分利用 GPU 算力
            'rect': True,                   # 【v5.0 支持】是否使用矩形训练（Rectangular Training）
                                            # 减少图像填充，提高训练速度，但可能轻微影响精度
            'label_smoothing': 0.05,        # 【v5.0 支持】标签平滑系数（0.0~0.1）
                                            # 例如 0.05 可防止过拟合，提升泛化能力。0.0 表示不启用

            # ----- GPU 显存（用于自动推荐模型）-----
            'gpu_memory': 8,                # GPU 显存大小（GB），用于 get_recommended_model()
                                            # RTX 4060 Laptop 为 8GB
        }

        # ==================== 优化器配置 ====================
        self.OPTIMIZER = {

            # ----- 优化器选择 -----
            'optimizer': 'SGD',             # 优化器类型（v6.0+ 支持 'SGD', 'Adam', 'AdamW'）
                                            # v5.0 通过 --adam 标志控制
            'use_adam': False,              # 【v5.0 支持】是否使用 Adam 优化器（True 则添加 --adam 参数）
                                            # 默认 SGD 收敛效果通常更好，Adam 可能更快但需调低学习率

            # ----- 学习率与动量 -----
            'lr0': 0.01,                    # 初始学习率（Initial Learning Rate）
                                            # 如果增大 batch_size，建议按比例提高 lr0
            'lrf': 0.2,                     # 最终学习率系数（Final LR factor），最终 lr = lr0 * lrf
            'momentum': 0.937,              # SGD 动量（Momentum），默认 0.937
            'weight_decay': 0.0005,         # 权重衰减系数（L2 正则化），防止过拟合

            # ----- 热身（Warmup）超参数 -----
            'warmup_epochs': 3.0,           # 热身轮数（Warmup Epochs），初期学习率从低到高增长
            'warmup_momentum': 0.8,         # 热身阶段的初始动量
            'warmup_bias_lr': 0.1,          # 热身阶段偏置项的学习率系数
        }

        # ==================== 数据增强配置 ====================
        # 这些参数会被写入 hyp.custom.yaml 文件（v5.0 通过 --hyp 传递）
        # 详细说明参考：https://github.com/ultralytics/yolov5/wiki/hyperparameters
        self.AUGMENTATION = {

            # ----- 色彩空间增强 -----
            'hsv_h': 0.015,                 # 色调（Hue）增强最大偏移量（比例）
            'hsv_s': 0.7,                   # 饱和度（Saturation）增强最大偏移量
            'hsv_v': 0.4,                   # 明度（Value）增强最大偏移量

            # ----- 几何变换增强 -----
            'degrees': 0.0,                 # 随机旋转角度（度），0 表示不旋转
            'translate': 0.1,               # 随机平移比例（相对于图像尺寸）
            'scale': 0.5,                   # 随机缩放比例（因子范围 1±scale）
            'shear': 0.0,                   # 随机剪切强度（度）
            'perspective': 0.0,             # 随机透视变换强度（因子），非零会降低性能

            # ----- 翻转增强 -----
            'flipud': 0.0,                  # 垂直翻转概率（0~1）
            'fliplr': 0.0,                  # 水平翻转概率（0~1），常用于一般目标检测

            # ----- 高级数据增强 -----
            'mosaic': 1.0,                  # Mosaic 数据增强概率（将 4 张图拼成 1 张）
            'mixup': 0.0,                   # MixUp 数据增强概率（v5.0 可能支持有限，建议保持 0）
        }

        # ==================== 预训练权重配置 ====================
        self.WEIGHTS = {
            'download_missing': True,       # 如果本地缺少预训练权重，是否自动下载
            'auto_download_yolov5': True,   # 如果本地缺少 YOLOv5 代码库，是否自动从 GitHub 下载
        }

        # ==================== 日志配置 ====================
        self.LOGGING = {
            'log_level': 'INFO',            # 日志级别：DEBUG, INFO, WARNING, ERROR
            'log_file': root / "training.log",  # 日志文件保存路径
        }

        # ---- 冻结配置，禁止运行时修改 ----
        self._freeze()

    def _freeze(self):
        """递归冻结所有内部 dict，防止运行时修改"""
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, types.MappingProxyType(value))
        self._frozen = True

    def __setattr__(self, name, value):
        """
        冻结保护：实例初始化完成后，禁止添加或修改任何非私有属性。
        使用 MappingProxyType 同时保护内部 dict 不被修改。
        """
        if getattr(self, '_frozen', False) and not name.startswith('_'):
            raise AttributeError(f"配置已冻结，不允许运行时修改属性: {name}")
        super().__setattr__(name, value)

    # ==================== 集中化工具方法 ====================

    def get_version_key(self) -> str:
        """
        版本号转路径键

        将配置中的版本号（如 '5.0'）转换为 PATHS 字典中使用的键名（如 'yolov5_5_0'）。
        此方法集中化，避免各模块重复编写相同的转换逻辑。

        Returns:
            路径键字符串，如 'yolov5_5_0'
        """
        return f"yolov5_{self.YOLO_VERSION.replace('.', '_')}"

    def get_yolov5_path(self) -> Path:
        """
        获取当前版本对应的 YOLOv5 代码库路径，作为 VersionManager.ensure() 的 local_dir。

        如果该路径存在，VersionManager 直接使用；否则下载到此处。

        Returns:
            代码库目录的 Path 对象，如 D:/auto_yolov5_train/yolov5/v5.0
        """
        return self.PATHS[self.get_version_key()]

    def get_yolov5_paths(self) -> list[tuple[str, Path]]:
        """
        列出项目目录中所有已存在的 YOLOv5 版本

        用于在日志中提示用户有哪些版本可选。

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
        根据 GPU 显存推荐合适的 YOLOv5 模型

        显存区间与推荐模型的对应关系：
            < 4GB   → yolov5s.pt（最小的模型，适合低端 GPU）
            4-8GB   → yolov5m.pt（中等大小，平衡速度与精度）
            8-12GB  → yolov5l.pt（大模型，高精度）
            ≥ 12GB  → yolov5x.pt（最大模型，最高精度）

        Note:
            原代码在此方法中存在 bug：8-12GB 区间误写为 yolov5s.pt（最小模型），
            已修正为 yolov5l.pt。

        Args:
            gpu_memory_gb: GPU 显存大小（GB）。为 None 时自动使用配置中的 gpu_memory 值。

        Returns:
            推荐模型的文件名，如 'yolov5l.pt'
        """
        mem = gpu_memory_gb if gpu_memory_gb is not None else self.TRAINING.get('gpu_memory', 8)
        if mem < 4:
            return 'yolov5s.pt'
        elif mem < 8:
            return 'yolov5m.pt'
        elif mem < 12:
            # 修复原 bug: 原来是 'yolov5s.pt'
            return 'yolov5l.pt'
        else:
            return 'yolov5x.pt'

    def check_version_compatibility(self, logger=None):
        """
        检查 YOLOv5 版本兼容性并打印信息

        根据版本号输出对应的训练参数配置方式：
            - v5.0 及以下：使用 hyp.yaml 文件 + 命令行标志（--multi-scale, --rect 等）
            - v6.0 及以上：使用命令行参数直接传递所有超参

        Args:
            logger: logging.Logger 实例，为 None 时静默返回

        Returns:
            bool: 始终返回 True
        """
        version = float(self.YOLO_VERSION)
        if logger is None:
            return True
        if version < 6.0:
            logger.info(f"YOLOv5 v{version}：使用 hyp.yaml 文件配置超参，--cache-images 缓存")
            if self.TRAINING.get('multi_scale'):
                logger.info("多尺度训练 (--multi-scale) 已开启，可充分利用 GPU 算力")
            if self.TRAINING.get('rect'):
                logger.info("矩形训练 (--rect) 已开启，将加速训练")
            if self.TRAINING.get('label_smoothing', 0.0) > 0:
                logger.info(f"标签平滑 (--label-smoothing {self.TRAINING['label_smoothing']}) 已启用")
        else:
            logger.info(f"YOLOv5 v{version}：使用命令行参数传递超参和增强配置")
        return True

    # ==================== 集中化超参序列化 ====================
    # 以下方法将配置字典转换为 CLI 参数或 YAML 内容，
    # 避免 v5.py / v6plus.py 中重复硬编码字段名。

    @staticmethod
    def _dict_to_cli_flags(d: dict) -> list[str]:
        """将配置 dict 转换为 CLI 参数列表"""
        result = []
        for key, value in d.items():
            result.append(f'--{key}')
            result.append(str(value))
        return result

    def generate_hyp_yaml_content(self) -> str:
        """生成 v5.0 兼容的 hyp.yaml 内容（动态从配置字典生成）"""
        lines = ["# YOLOv5 Hyperparameters (generated by config)"]
        lines.append("")
        lines.append("# Optimizer")
        for key, value in self.OPTIMIZER.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("# Augmentation")
        for key, value in self.AUGMENTATION.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("# Loss (fixed values)")
        for key, value in [
            ("box", 0.05), ("cls", 0.5), ("cls_pw", 1.0),
            ("obj", 1.0), ("obj_pw", 1.0), ("iou_t", 0.20),
            ("anchor_t", 4.0), ("fl_gamma", 0.0),
        ]:
            lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"

    def get_optimizer_cli_args(self) -> list[str]:
        """获取优化器 CLI 参数列表（v6.0+）"""
        return self._dict_to_cli_flags(self.OPTIMIZER)

    def get_augmentation_cli_args(self) -> list[str]:
        """获取数据增强 CLI 参数列表（v6.0+）"""
        return self._dict_to_cli_flags(self.AUGMENTATION)
