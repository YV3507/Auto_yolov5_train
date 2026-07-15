"""
v6.0+ 版本策略

训练参数传递方式：
  - 所有超参通过命令行参数直接传递
  - 使用 --cache ram 缓存数据到内存
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import VersionStrategy

logger = logging.getLogger('yolov5_trainer')


class V6PlusStrategy(VersionStrategy):
    """v6.0 及以上版本策略（v6.0/v7.0 共用）"""

    def build_command(
        self,
        config,
        yolov5_path: Path,
        weights_path: Optional[str],
    ) -> list[str]:
        cmd = self._build_base_command(config, yolov5_path, weights_path)

        cmd.extend(['--patience', str(config.TRAINING['patience'])])
        cmd.extend(self._get_optimizer_args(config))
        cmd.extend(self._get_augmentation_args(config))
        cmd.extend(['--cache', 'ram'])

        logger.info(f"YOLOv5 v{config.YOLO_VERSION}：使用命令行参数配置训练")
        return cmd

    # ==================== 参数辅助方法 ====================

    def _get_optimizer_args(self, config) -> list[str]:
        """获取优化器参数列表"""
        o = config.OPTIMIZER
        return [
            '--optimizer', o['optimizer'],
            '--lr0', str(o['lr0']),
            '--lrf', str(o['lrf']),
            '--momentum', str(o['momentum']),
            '--weight_decay', str(o['weight_decay']),
            '--warmup_epochs', str(o['warmup_epochs']),
            '--warmup_momentum', str(o['warmup_momentum']),
            '--warmup_bias_lr', str(o['warmup_bias_lr']),
        ]

    def _get_augmentation_args(self, config) -> list[str]:
        """获取数据增强参数列表"""
        a = config.AUGMENTATION
        return [
            '--hsv_h', str(a['hsv_h']),
            '--hsv_s', str(a['hsv_s']),
            '--hsv_v', str(a['hsv_v']),
            '--degrees', str(a['degrees']),
            '--translate', str(a['translate']),
            '--scale', str(a['scale']),
            '--shear', str(a['shear']),
            '--perspective', str(a['perspective']),
            '--flipud', str(a['flipud']),
            '--fliplr', str(a['fliplr']),
            '--mosaic', str(a['mosaic']),
            '--mixup', str(a['mixup']),
        ]
