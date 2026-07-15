"""
YOLOv5 / Ultralytics 版本策略注册
通过 STRATEGY_MAP 实现版本/模型 → 策略的多态分发，无需 if/else 分支。

策略键说明：
  - YOLOv5 体系：以版本号为键（如 "5.0", "6.0", "7.0"），因其有真实子版本
  - Ultralytics 体系：以模型名为键（如 "yolov8"），因其无子版本，模型名即标识

使用约定：
  - 添加新版本时只需在 STRATEGY_MAP 中添加一行，
    同时在对应配置类的 ALLOWED_VERSIONS / ALLOWED_MODELS 中添加对应项。
"""

from __future__ import annotations

from .v5 import V5Strategy
from .v6plus import V6PlusStrategy
from .ultralytics import UltralyticsStrategy

# 版本/模型 → 策略类映射（开闭原则：加新版本只需加一行）
STRATEGY_MAP: dict[str, type] = {
    # YOLOv5 体系（有子版本）
    "5.0": V5Strategy,
    "6.0": V6PlusStrategy,
    "7.0": V6PlusStrategy,
    # Ultralytics 体系（无子版本，模型名即标识）
    "yolov8": UltralyticsStrategy,
    "yolov10": UltralyticsStrategy,
    "yolo11": UltralyticsStrategy,
}


def get_strategy(identifier: str):
    """根据版本号或模型名获取对应的策略实例"""
    cls = STRATEGY_MAP.get(identifier)
    if not cls:
        raise ValueError(
            f"不支持的版本/模型: {identifier}，可选: {list(STRATEGY_MAP.keys())}"
        )
    return cls()
