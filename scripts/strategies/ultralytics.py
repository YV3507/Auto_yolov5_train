"""
Ultralytics 体系训练策略

将 UltralyticsConfig 配置转换为 `yolo train key=value` CLI 命令。
与 YOLOv5 体系策略并列，通过 STRATEGY_MAP 注册。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .base import VersionStrategy


class UltralyticsStrategy(VersionStrategy):
    """Ultralytics 体系训练策略（YOLOv8/v10/v11）"""

    def build_command(
        self,
        config,
        yolov5_path: Optional[Path] = None,  # Ultralytics 无需 YOLOv5 源码
        weights_path: Optional[str] = None,
    ) -> list[str]:
        """
        构建 Ultralytics yolo train CLI 命令

        Args:
            config: UltralyticsConfig 实例
            yolov5_path: 忽略（Ultralytics 不需要 YOLOv5 源码）
            weights_path: 权重的完整路径或模型名称，如 "yolov8n.pt"

        Returns:
            CLI 命令列表，供 subprocess.Popen 执行
        """
        cmd = [sys.executable, '-m', 'ultralytics.cfg', 'yolo', 'train']

        # 模型参数：如有 weights_path 则覆盖配置中的 model
        model_name = weights_path or config.TRAINING['model']
        cmd.append(f'model={model_name}')

        # 其余 CLI 参数由配置生成
        cmd.extend(config.build_cli_args())

        return cmd
