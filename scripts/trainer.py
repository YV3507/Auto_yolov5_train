"""
YOLOv5 训练执行模块

职责变更（重构后）：
  - 不再负责版本代码下载/缓存管理 → 由 VersionManager 处理
  - 不再负责版本分支逻辑 → 由 VersionStrategy 多态分发
  - 只负责：信号处理 + subprocess.Popen 执行

改进：
  - subprocess.Popen(cwd=...) 替代 os.chdir，消除工作目录副作用
  - 信号处理器只注册一次（模块级 _handlers_registered 标志）
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

logger = logging.getLogger('yolov5_trainer')

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


# ==================== 训练执行 ====================


def run_training(config, weights_path: Optional[str] = None) -> bool:
    """
    执行 YOLOv5 训练

    流程：
        1. VersionManager.ensure() → 获取/缓存版本代码路径
        2. get_strategy().build_command() → 多态构建命令
        3. subprocess.Popen(cwd=...) → 执行训练

    Args:
        config: YOLOv5Config 实例
        weights_path: 预训练权重路径，为 None 时从头训练

    Returns:
        True 表示训练成功，False 表示失败
    """
    global _training_process

    logger.info("开始 YOLOv5 训练...")

    # 1. 确保版本代码可用（本地优先 → 缓存兜底）
    try:
        vman = VersionManager()
        local_dir = config.get_yolov5_path() if config.PATHS.get(config.get_version_key()) else None
        yolov5_path = vman.ensure(config.YOLO_VERSION, local_dir=local_dir)
    except RuntimeError as e:
        logger.error(f"YOLOv5 代码库获取失败: {e}")
        return False

    # 2. 构建训练命令（多态分发，无 if/else）
    strategy = get_strategy(config.YOLO_VERSION)
    cmd = strategy.build_command(config, yolov5_path, weights_path)
    logger.info(f"训练命令: {' '.join(cmd)}")

    # 3. 执行
    _register_signal_handlers()

    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['WANDB_DIR'] = str(config.PATHS.get('wandb_dir', config.PROJECT_ROOT / "wandb"))
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    _training_process = subprocess.Popen(
        cmd,
        cwd=str(yolov5_path),
        env=env,
    )
    return_code = _training_process.wait()
    _training_process = None

    if return_code == 0:
        logger.info("=== 训练成功完成 ===")
        return True
    else:
        logger.error(f"=== 训练失败，返回码: {return_code} ===")
        return False
