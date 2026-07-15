"""
Ultralytics 包与环境管理

管理 ultralytics pip 包的安装检查和兼容性验证。
Ultralytics 的权重由包在首次使用 YOLO(model) 时自动下载，无需单独实现下载逻辑。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Ultralytics 所需的最低 pip 包版本
ULTRALYTICS_MIN_VERSION = "8.0.0"


class UltralyticsManager:
    """管理 ultralytics 包安装和环境"""

    @staticmethod
    def ensure_installed(package_spec: Optional[str] = None) -> bool:
        """
        检查 ultralytics 是否安装，未安装则自动 pip install

        Args:
            package_spec: 可选，指定安装版本约束（如 ">=8.3.0"）

        Returns:
            True 表示安装/可用，False 表示安装失败
        """
        try:
            import ultralytics  # noqa: F401
            logger.info(f"ultralytics 包已安装 (v{ultralytics.__version__})")
            return True
        except ImportError:
            logger.info("ultralytics 包未安装，正在自动安装...")

        pip_cmd = [sys.executable, "-m", "pip", "install", "ultralytics"]
        if package_spec:
            pip_cmd.append(package_spec)

        try:
            subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
            logger.info("ultralytics 包安装成功")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"ultralytics 安装失败: {e.stderr}")
            return False

    @staticmethod
    def get_model_name(model: str, model_size: str = 'n') -> str:
        """
        获取 Ultralytics 体系下的模型文件名

        Args:
            model: 模型名，如 "yolov8", "yolo11"
            model_size: 模型大小，如 "n", "s", "m", "l", "x"

        Returns:
            模型文件名，如 "yolov8n.pt", "yolo11s.pt"
        """
        from scripts.config import UltralyticsConfig
        return UltralyticsConfig.get_model_name(model, model_size)

    @staticmethod
    def check_version_compatibility(model: str) -> bool:
        """
        检查 ultralytics 包版本是否支持所请求的 YOLO 模型

        Args:
            model: YOLO 模型名，如 "yolov8", "yolo11"

        Returns:
            True 表示兼容
        """
        try:
            import ultralytics
            pkg_version = ultralytics.__version__
            pkg_major = int(pkg_version.split(".")[0])
            # ultralytics 8.x 包支持 yolov8/yolov10/yolo11 等模型
            if pkg_major < 8:
                logger.warning(
                    f"ultralytics v{pkg_version} 可能不支持 {model}，"
                    f"建议升级到最新版本"
                )
            return True
        except ImportError:
            return False
