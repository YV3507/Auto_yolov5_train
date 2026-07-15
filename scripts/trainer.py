"""
YOLOv5 / Ultralytics 训练执行模块

职责：
  - 体系检测（YOLOv5 vs Ultralytics）—— 使用 isinstance 而非版本号比较
  - 信号处理 + subprocess.Popen 执行

改进：
  - subprocess.Popen(cwd=...) 替代 os.chdir，消除工作目录副作用
  - 信号处理器只注册一次（模块级 _handlers_registered 标志）
  - 双体系并行支持，互不干扰
  - 体系检测基于配置类类型，而非浮点数版本比较
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from typing import Optional

from scripts.strategies import get_strategy
from scripts.version_manager import VersionManager
from scripts.ultralytics_manager import UltralyticsManager

logger = logging.getLogger(__name__)

# ==================== 信号处理（只注册一次） ====================

_handlers_registered = False
_training_process: Optional[subprocess.Popen] = None


def _register_signal_handlers():
    """注册 SIGINT/SIGTERM 处理器，多次调用仅首次生效"""
    global _handlers_registered
    if _handlers_registered:
        return

    def handler(sig, frame):
        global _training_process
        print("\n收到中断信号，正在停止训练...")
        if _training_process and _training_process.poll() is None:
            _training_process.terminate()
            try:
                _training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _training_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    _handlers_registered = True


# ==================== 体系检测 ====================


def _is_ultralytics_config(config) -> bool:
    """通过配置类类型判断是否属于 Ultralytics 体系"""
    from scripts.config import UltralyticsConfig
    return isinstance(config, UltralyticsConfig)


# ==================== 公共环境变量 ====================


def _prepare_env(config) -> dict[str, str]:
    """准备子进程环境变量（WANDB / UTF-8 编码）"""
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['WANDB_DIR'] = str(config.PATHS.get('wandb_dir', config.PROJECT_ROOT / "wandb"))
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


# ==================== 训练执行 ====================


def run_training(config, weights_path: Optional[str] = None) -> bool:
    """
    执行训练（YOLOv5 或 Ultralytics 体系）

    Args:
        config: YOLOv5Config 或 UltralyticsConfig 实例
        weights_path: 预训练权重路径，为 None 时从头训练

    Returns:
        True 表示训练成功，False 表示失败
    """
    global _training_process

    if _is_ultralytics_config(config):
        return _run_ultralytics_training(config, weights_path)
    else:
        return _run_yolov5_training(config, weights_path)


def _run_yolov5_training(config, weights_path: Optional[str] = None) -> bool:
    """执行 YOLOv5 体系训练"""
    global _training_process

    logger.info("开始 YOLOv5 训练...")

    # 1. 确保版本代码可用（本地优先 → 缓存兜底）
    try:
        vman = VersionManager()
        yolov5_local = config.get_yolov5_path()
        # 仅在目录实际存在时才作为 local_dir 传入，否则走缓存兜底
        local_dir = yolov5_local if yolov5_local.exists() else None
        yolov5_path = vman.ensure(config.YOLO_VERSION, local_dir=local_dir)
    except RuntimeError as e:
        logger.error(f"YOLOv5 代码库获取失败: {e}")
        return False

    # 2. 构建训练命令（多态分发）
    strategy = get_strategy(config.YOLO_VERSION)
    cmd = strategy.build_command(config, yolov5_path, weights_path)
    logger.info(f"YOLOv5 训练命令: {' '.join(cmd)}")

    # 3. 执行
    _register_signal_handlers()
    env = _prepare_env(config)

    _training_process = subprocess.Popen(
        cmd,
        cwd=str(yolov5_path),
        env=env,
    )
    return_code = _training_process.wait()
    _training_process = None

    if return_code == 0:
        logger.info("=== YOLOv5 训练成功完成 ===")
        return True
    else:
        logger.error(f"=== YOLOv5 训练失败，返回码: {return_code} ===")
        return False


def _run_ultralytics_training(config, weights_path: Optional[str] = None) -> bool:
    """执行 Ultralytics 体系训练"""
    global _training_process

    model_display = config.MODEL
    logger.info(f"开始 {model_display} 训练...")

    # 1. 确保 ultralytics 包已安装
    if not UltralyticsManager.ensure_installed():
        logger.error("ultralytics 包安装失败")
        return False

    UltralyticsManager.check_version_compatibility(config.MODEL)

    # 2. 构建训练命令
    strategy = get_strategy(config.MODEL)
    cmd = strategy.build_command(config, yolov5_path=None, weights_path=weights_path)
    logger.info(f"{model_display} 训练命令: {' '.join(cmd)}")

    # 3. 执行
    _register_signal_handlers()
    env = _prepare_env(config)

    _training_process = subprocess.Popen(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        env=env,
    )
    return_code = _training_process.wait()
    _training_process = None

    if return_code == 0:
        logger.info(f"=== {model_display} 训练成功完成 ===")
        return True
    else:
        logger.error(f"=== {model_display} 训练失败，返回码: {return_code} ===")
        return False
