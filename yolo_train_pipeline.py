#!/usr/bin/env python3
"""
YOLO 自动化训练管道主脚本

支持 YOLOv5 / YOLOv8 / YOLOv10 / YOLO11 多体系训练，
通过统一的 CLI 接口分发到对应版本的训练引擎。

使用带前缀的版本/模型标识避免混淆：
  - YOLOv5 体系：yolov5-5.0, yolov5-6.0, yolov5-7.0（有子版本号）
  - Ultralytics 体系：yolov8, yolov10, yolo11（无子版本号，模型名即标识）

内部标识规则（与用户标识不同）：
  - YOLOv5：版本号字符串 "5.0", "6.0", "7.0"（反映真实子版本）
  - Ultralytics：模型名字符串 "yolov8", "yolo11"（无子版本，模型名即标识）

使用方法：
    python yolo_train_pipeline.py --yolo-version yolov5-5.0
    python yolo_train_pipeline.py --yolo-version yolov8
    python yolo_train_pipeline.py --yolo-version yolo11
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ==================== 默认版本配置 ====================
# 可通过环境变量 YOLO_DEFAULT_VERSION 覆盖，方便 CI/CD 场景
DEFAULT_YOLO_VERSION = os.environ.get('YOLO_DEFAULT_VERSION', 'yolov5-5.0')

# ==================== 标识映射 ====================
# Ultralytics 模型的内部标识就是模型名本身，无需间接映射
ULTRALYTICS_MODELS = {"yolov8", "yolov10", "yolo11"}

# 用户友好标识 → 内部标识
# YOLOv5 内部为版本号（如 "5.0"），Ultralytics 内部为模型名（如 "yolov8"）
YOLO_VERSION_MAP = {
    # YOLOv5 体系（有子版本）
    'yolov5-5.0': '5.0',
    'yolov5-6.0': '6.0',
    'yolov5-7.0': '7.0',
    # Ultralytics 体系（无子版本，内部标识即模型名）
    'yolov8': 'yolov8',
    'yolov10': 'yolov10',
    'yolo11': 'yolo11',
}

# 内部标识 → 用户友好显示名的反向映射
_INTERNAL_TO_DISPLAY: dict[str, str] = {
    '5.0': 'yolov5-5.0',
    '6.0': 'yolov5-6.0',
    '7.0': 'yolov5-7.0',
    'yolov8': 'yolov8',
    'yolov10': 'yolov10',
    'yolo11': 'yolo11',
}


def parse_yolo_version(raw: str) -> str:
    """解析用户输入的版本/模型标识，返回内部标识"""
    if raw not in YOLO_VERSION_MAP:
        primary_choices = ['yolov5-5.0', 'yolov5-6.0', 'yolov5-7.0',
                           'yolov8', 'yolov10', 'yolo11']
        raise ValueError(
            f"不支持的版本标识: {raw}，可选: {primary_choices}"
        )
    return YOLO_VERSION_MAP[raw]


def _user_friendly_id(internal_id: str) -> str:
    """将内部标识转为用户友好的显示名称（O(1) 查表）"""
    return _INTERNAL_TO_DISPLAY.get(internal_id, f"YOLOv5 v{internal_id}")


def create_config(identifier: str):
    """
    根据内部标识创建对应的配置实例

    Args:
        identifier: 内部标识
            YOLOv5：版本号，如 "5.0", "6.0", "7.0"
            Ultralytics：模型名，如 "yolov8", "yolo11"

    Returns:
        YOLOv5Config 或 UltralyticsConfig 实例
    """
    if identifier in ULTRALYTICS_MODELS:
        from scripts.config import UltralyticsConfig
        return UltralyticsConfig(model=identifier)
    else:
        from scripts.config import YOLOv5Config
        return YOLOv5Config(version=identifier)


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """设置日志系统"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
    )
    return logging.getLogger('yolov5_trainer')


def log_config(cfg, log: logging.Logger, display_name: str) -> None:
    """打印配置概览（兼容两种配置类）"""
    model_id = getattr(cfg, 'MODEL', cfg.YOLO_VERSION)
    log.info(f"YOLO 标识: {model_id} ({display_name})")
    log.info(f"项目根目录: {cfg.PROJECT_ROOT}")
    log.info(f"设备: {cfg.TRAINING.get('device', '0')}")
    # 统一使用 batch / imgsz（两 Config 类均已统一字段名）
    log.info(f"Batch Size: {cfg.TRAINING['batch']}")
    log.info(f"图像尺寸: {cfg.TRAINING['imgsz']}")
    log.info(f"训练轮数: {cfg.TRAINING['epochs']}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 自动化训练管道（YOLOv5 / YOLOv8 / YOLOv10 / YOLO11）"
    )
    parser.add_argument(
        '--yolo-version', type=str, default=DEFAULT_YOLO_VERSION,
        choices=['yolov5-5.0', 'yolov5-6.0', 'yolov5-7.0', 'yolov8', 'yolov10', 'yolo11'],
        help=(
            f'YOLO 版本标识（默认: {DEFAULT_YOLO_VERSION}）。'
            'YOLOv5 体系: yolov5-5.0 / yolov5-6.0 / yolov5-7.0；'
            'Ultralytics 体系: yolov8 / yolov10 / yolo11'
        ),
    )
    args = parser.parse_args()

    # 解析为用户友好的显示名和内部标识
    raw = args.yolo_version
    internal_id = parse_yolo_version(raw)
    display_name = _user_friendly_id(internal_id)
    is_ultra = internal_id in ULTRALYTICS_MODELS

    try:
        # 1. 初始化配置
        cfg = create_config(internal_id)

        # 2. 设置日志
        log_file = cfg.PATHS.get('log_file', cfg.PROJECT_ROOT / "training.log")
        log_level = cfg.LOGGING.get('log_level', 'INFO') if hasattr(cfg, 'LOGGING') else 'INFO'
        log = setup_logging(log_file, log_level)
        log.info(f"=== YOLO 自动化训练管道启动 ({display_name}) ===")
        log_config(cfg, log, display_name)

        # 3. 创建目录
        from scripts.data import create_directories
        create_directories([
            cfg.PATHS['raw_data'],
            cfg.PATHS['images_dir'],
            cfg.PATHS['labels_dir'],
            cfg.PATHS['project_dir'],
            cfg.PATHS['weights_dir'],
        ])
        log.info("目录检查/创建完成")

        # 4. 准备训练数据
        from scripts.data import prepare_data
        prepare_data(
            images_dir=cfg.PATHS['images_dir'],
            labels_dir=cfg.PATHS['labels_dir'],
            classes_file=cfg.PATHS['classes_file'],
            project_dir=cfg.PATHS['project_dir'],
            annotation_format=cfg.DATA['annotation_format'],
            train_ratio=cfg.DATA['train_ratio'],
            val_ratio=cfg.DATA['val_ratio'],
            test_ratio=cfg.DATA['test_ratio'],
            random_seed=cfg.DATA['random_seed'],
        )

        # 5. 确定预训练模型（get_recommended_model 内部已处理用户指定 vs 自动推荐）
        recommended_model = cfg.get_recommended_model()
        log.info(f"使用模型: {recommended_model}")

        from scripts.version_manager import ensure_weights, ensure_ultralytics_weights
        if is_ultra:
            weights_path = ensure_ultralytics_weights(
                recommended_model,
                weights_dir=cfg.PATHS['weights_dir'],
            )
        else:
            weights_path = ensure_weights(
                model_type=recommended_model,
                version=internal_id,
                weights_dir=cfg.PATHS['weights_dir'],
                download_missing=cfg.WEIGHTS['download_missing'],
            )

        if weights_path:
            log.info(f"使用预训练权重: {weights_path}")
        else:
            log.info("从头开始训练（无预训练权重）")

        # 6. 执行训练
        from scripts.trainer import run_training
        success = run_training(cfg, weights_path)

        if success:
            print(f"\n训练完成！({display_name})")
            print(f"结果保存在: {cfg.PATHS['project_dir'] / 'train_results'}")
        else:
            print(f"\n训练失败 ({display_name})，请检查日志")
            sys.exit(1)

    except Exception as e:
        print(f"训练管道发生严重错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
