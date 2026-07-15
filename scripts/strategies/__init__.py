"""
YOLOv5 版本策略注册
通过 STRATEGY_MAP 实现版本 → 策略的多态分发，无需 if/else 分支。
"""

from __future__ import annotations

from .base import VersionStrategy
from .v5 import V5Strategy
from .v6plus import V6PlusStrategy

# 版本 → 策略类映射（开闭原则：加新版本只需加一行）
STRATEGY_MAP: dict[str, type[VersionStrategy]] = {
    "5.0": V5Strategy,
    "6.0": V6PlusStrategy,
    "7.0": V6PlusStrategy,
}


def get_strategy(version: str) -> VersionStrategy:
    """根据版本号获取对应的策略实例"""
    cls = STRATEGY_MAP.get(version)
    if not cls:
        raise ValueError(
            f"不支持的 YOLOv5 版本: {version}，可选: {list(STRATEGY_MAP.keys())}"
        )
    return cls()
