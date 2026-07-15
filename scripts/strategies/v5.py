"""
v5.0 及以下版本策略

训练参数传递方式：
  - 通过 hyp.yaml 文件传递超参（优化器 + 数据增强）
  - 通过命令行标志控制功能开关（multi-scale, rect, label-smoothing, adam）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import VersionStrategy

logger = logging.getLogger('yolov5_trainer')


class V5Strategy(VersionStrategy):
    """v5.0 及以下版本策略"""

    def build_command(
        self,
        config,
        yolov5_path: Path,
        weights_path: Optional[str],
    ) -> list[str]:
        cmd = self._build_base_command(config, yolov5_path, weights_path)

        # hyp 文件
        hyp_file = self._create_hyp_file(config, yolov5_path)
        if hyp_file:
            cmd.extend(['--hyp', str(hyp_file)])

        cmd.append('--cache-images')

        if config.TRAINING.get('multi_scale', False):
            cmd.append('--multi-scale')
        if config.TRAINING.get('rect', False):
            cmd.append('--rect')
        ls = config.TRAINING.get('label_smoothing', 0.0)
        if ls > 0:
            cmd.extend(['--label-smoothing', str(ls)])
        if config.OPTIMIZER.get('use_adam', False):
            cmd.append('--adam')

        logger.info(f"YOLOv5 v{config.YOLO_VERSION}：使用 hyp.yaml + 标志参数配置训练")
        return cmd

    # ==================== hyp 文件管理 ====================

    def _create_hyp_file(self, config, yolov5_path: Path) -> Optional[Path]:
        """创建或定位 hyp 文件"""
        default_hyp = yolov5_path / "data" / "hyp.scratch.yaml"
        if default_hyp.exists():
            logger.info(f"使用 YOLOv5 默认 hyp 文件: {default_hyp}")
            return default_hyp

        hyp_file = config.PATHS['project_dir'] / 'hyp.custom.yaml'
        hyp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hyp_file, 'w', encoding='utf-8') as f:
            f.write(config.generate_hyp_yaml_content())
        logger.info(f"创建自定义 hyp 文件: {hyp_file}")
        return hyp_file
