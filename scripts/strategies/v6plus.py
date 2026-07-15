"""
YOLOv5 v6.0+ 版本策略

训练参数传递方式：
  - 所有超参通过命令行参数直接传递
  - 使用 --cache ram 缓存数据到内存
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import VersionStrategy

logger = logging.getLogger(__name__)


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
        cmd.extend(self._dict_to_cli_args(config.OPTIMIZER))
        cmd.extend(self._dict_to_cli_args(config.AUGMENTATION))
        cmd.extend(['--cache', 'ram'])

        logger.info(f"YOLOv5 v{config.YOLO_VERSION}：使用命令行参数配置训练")
        return cmd

    @staticmethod
    def _dict_to_cli_args(d: dict) -> list[str]:
        """将配置 dict 转换为 --key value 格式的 CLI 参数列表"""
        result = []
        for key, value in d.items():
            result.append(f'--{key}')
            result.append(str(value))
        return result
