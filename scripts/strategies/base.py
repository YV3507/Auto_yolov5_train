"""
YOLOv5 版本策略抽象基类

每个 YOLOv5 版本构造训练命令的方式不同：
  - v5.0 及以下：hyp.yaml + 命令行标志参数
  - v6.0+：命令行参数直接传递所有超参

子类只需实现 build_command()，公共方法集中于此。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger('yolov5_trainer')


class VersionStrategy(ABC):
    """YOLOv5 版本策略抽象基类"""

    @abstractmethod
    def build_command(
        self,
        config,
        yolov5_path: Path,
        weights_path: Optional[str],
    ) -> list[str]:
        """构建训练命令列表，供 subprocess.Popen 执行"""
        ...

    # ==================== 公共工具方法 ====================

    def _get_model_config_path(self, yolov5_path: Path, model_config: str) -> str:
        """获取模型配置文件（如 yolov5s.yaml）的绝对路径"""
        models_dir = yolov5_path / "models"
        for path in [models_dir / model_config, yolov5_path / model_config, Path(model_config)]:
            if path.exists():
                return str(path)
        default = models_dir / "yolov5s.yaml"
        logger.warning(f"未找到模型配置 {model_config}，使用默认 {default}")
        return str(default)

    def _build_base_command(
        self,
        config,
        yolov5_path: Path,
        weights_path: Optional[str],
    ) -> list[str]:
        """构建所有版本共用的基础命令参数"""
        import sys
        cmd = [
            sys.executable, "train.py",
            '--img', str(config.TRAINING['img_size']),
            '--batch', str(config.TRAINING['batch_size']),
            '--epochs', str(config.TRAINING['epochs']),
            '--data', str(config.PATHS['project_dir'] / 'data.yaml'),
            '--cfg', self._get_model_config_path(yolov5_path, config.TRAINING['model']),
            '--device', config.TRAINING['device'],
            '--workers', str(config.TRAINING['workers']),
            '--project', str(config.PATHS['project_dir']),
            '--name', 'train_results',
            '--exist-ok',
        ]
        if weights_path:
            cmd.extend(['--weights', weights_path])
        return cmd
