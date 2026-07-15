#!/usr/bin/env python3
"""
YOLOv5 自动化训练管道主脚本
重构版本：无类封装、顺序调用、无全局可变状态、无循环依赖
"""

import logging
import sys
from pathlib import Path

from scripts.config import YOLOv5Config
from scripts.data import create_directories, prepare_data
from scripts.version_manager import ensure_weights
from scripts.trainer import run_training


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


def log_config(cfg: YOLOv5Config, log: logging.Logger) -> None:
    """打印配置概览"""
    log.info(f"YOLOv5 版本: {cfg.YOLO_VERSION}")
    log.info(f"项目根目录: {cfg.PROJECT_ROOT}")
    log.info(f"设备: {cfg.TRAINING['device']}")
    log.info(f"Batch Size: {cfg.TRAINING['batch_size']}")
    log.info(f"图像尺寸: {cfg.TRAINING['img_size']}")
    log.info(f"训练轮数: {cfg.TRAINING['epochs']}")


def main():
    try:
        # 1. 初始化配置
        cfg = YOLOv5Config(version="5.0")

        # 2. 设置日志
        log = setup_logging(cfg.LOGGING['log_file'], cfg.LOGGING['log_level'])
        log.info("=== YOLOv5 自动化训练管道启动 ===")
        log_config(cfg, log)
        cfg.check_version_compatibility(log)

        # 3. 创建目录
        create_directories([
            cfg.PATHS['raw_data'],
            cfg.PATHS['images_dir'],
            cfg.PATHS['labels_dir'],
            cfg.PATHS['project_dir'],
            cfg.PATHS['weights_dir'],
        ])
        log.info("目录检查/创建完成")

        # 4. 准备训练数据
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

        # 5. 获取推荐模型和预训练权重
        #    注意：YOLOv5 代码库由 VersionManager 在 run_training 内部自动缓存
        recommended_model = cfg.get_recommended_model()
        log.info(f"使用模型: {recommended_model}")

        weights_path = ensure_weights(
            model_type=recommended_model,
            version=cfg.YOLO_VERSION,
            weights_dir=cfg.PATHS['weights_dir'],
            download_missing=cfg.WEIGHTS['download_missing'],
        )

        if weights_path:
            log.info(f"使用预训练权重: {weights_path}")
        else:
            log.info("从头开始训练（无预训练权重）")

        # 6. 执行训练
        #    run_training 内部自动处理版本代码缓存和命令构建
        success = run_training(cfg, weights_path)

        if success:
            print("\n训练完成！")
            print(f"结果保存在: {cfg.PATHS['project_dir'] / 'train_results'}")
        else:
            print("\n训练失败，请检查日志")
            sys.exit(1)

    except Exception as e:
        print(f"训练管道发生严重错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
