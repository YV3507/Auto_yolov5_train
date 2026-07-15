"""
YOLOv5 版本和权重资源管理

双模式设计：
  - 本地优先：如果 local_dir 存在（如 project/yolov5/v5.0/），直接使用
  - 缓存兜底：local_dir 不存在/未指定时，下载到 ~/.cache/yolov5/v{version}/
  - 下载方式从 ZIP 下载 + 解压改为 git shallow clone

使用方法：
    from scripts.version_manager import VersionManager, ensure_weights

    # 模式一：本地优先
    vman = VersionManager()
    path = vman.ensure("5.0", local_dir=project_root/"yolov5"/"v5.0")

    # 模式二：自动缓存
    path = vman.ensure("5.0")               # 下载到 ~/.cache/yolov5/v5.0/

    vman.list_cached()                       # 列出已缓存版本
    vman.clean("5.0")                        # 清理指定缓存版本

    ensure_weights("yolov5s.pt", "5.0")      # 下载权重
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/ultralytics/yolov5.git"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "yolov5"


# ==================== 权重下载信息 ====================

# 每个模型支持的版本列表（URL 按规则自动生成）
AVAILABLE_WEIGHTS: dict[str, list[str]] = {
    'yolov5s.pt': ['5.0', '6.0', '6.1', '7.0'],
    'yolov5m.pt': ['5.0', '6.0', '7.0'],
    'yolov5l.pt': ['5.0', '6.0', '7.0'],
    'yolov5x.pt': ['5.0', '6.0', '7.0'],
}


def _generate_weight_url(model_type: str, version: str) -> str:
    """根据模型类型和版本生成下载 URL"""
    return f"https://github.com/ultralytics/yolov5/releases/download/v{version}/{model_type}"

WEIGHTS_CACHE_DIR = DEFAULT_CACHE_ROOT / "weights"


# ==================== 版本缓存管理 ====================


class VersionManager:
    """YOLOv5 版本代码缓存管理器（git shallow clone）"""

    def __init__(self, cache_root: Optional[Path] = None):
        self.cache_root = cache_root or DEFAULT_CACHE_ROOT

    def _version_dir(self, version: str) -> Path:
        return self.cache_root / f"v{version}"

    def ensure(self, version: str, local_dir: Optional[Path] = None) -> Path:
        """
        确保 YOLOv5 版本代码可用，返回有效路径。

        下载目标逻辑：
          - 指定了 local_dir → 下载到 local_dir（不存在则创建）
          - 未指定 local_dir → 下载到 ~/.cache/yolov5/v{version}/

        Args:
            version: 版本号，如 "5.0", "6.0", "7.0"
            local_dir: 本地目录路径（可选），指定后作为下载目标

        Returns:
            版本代码目录的 Path 对象
        """
        target = local_dir or self._version_dir(version)
        if target.exists():
            logger.info(f"YOLOv5 v{version} 已存在: {target}")
            return target

        logger.info(f"正在下载 YOLOv5 v{version} 到 {target} ...")
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", f"v{version}",
                 REPO_URL, str(target)],
                check=True, capture_output=True, text=True,
            )
            logger.info(f"YOLOv5 v{version} 下载完成")
            return target
        except subprocess.CalledProcessError as e:
            logger.error(f"git clone 失败: {e.stderr}")
            raise RuntimeError(f"下载 YOLOv5 v{version} 失败") from e

    def list_cached(self) -> list[tuple[str, Path]]:
        """列出所有已缓存的版本，按版本号排序"""
        if not self.cache_root.exists():
            return []
        result = []
        for d in self.cache_root.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                version = d.name[1:]  # 去掉 "v" 前缀
                result.append((version, d))
        return sorted(result, key=lambda x: tuple(map(int, x[0].split("."))))

    def clean(self, version: str) -> None:
        """删除指定版本的缓存"""
        target = self._version_dir(version)
        if target.exists():
            import shutil
            shutil.rmtree(target)
            logger.info(f"已清理 YOLOv5 v{version} 缓存")

    def clean_all(self) -> None:
        """清理所有 YOLOv5 缓存"""
        import shutil
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)
            logger.info("已清理所有 YOLOv5 缓存")


# ==================== 权重资源管理 ====================


def _download_with_progress(url: str, dest: Path, desc: str) -> None:
    """带进度条的文件下载"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))

    with open(dest, 'wb') as f, tqdm(
        desc=desc, total=total, unit='iB', unit_scale=True, unit_divisor=1024,
    ) as pb:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pb.update(len(chunk))


def _parse_version(v: str) -> tuple[int, ...]:
    """将版本号字符串解析为数值元组，用于语义化比较"""
    return tuple(int(x) for x in v.split("."))


def _find_download_url(model_type: str, version: str) -> Optional[str]:
    """
    查找最适合的权重下载 URL

    策略：精确版本匹配 → 最近的高版本 fallback → 最大版本
    """
    available = AVAILABLE_WEIGHTS.get(model_type)
    if not available:
        return None
    if version in available:
        return _generate_weight_url(model_type, version)
    sorted_versions = sorted(available, key=_parse_version)
    parsed_version = _parse_version(version)
    for v in sorted_versions:
        if _parse_version(v) >= parsed_version:
            return _generate_weight_url(model_type, v)
    return _generate_weight_url(model_type, sorted_versions[-1]) if sorted_versions else None


def ensure_weights(
    model_type: str,
    version: str,
    weights_dir: Optional[Path] = None,
    download_missing: bool = True,
) -> Optional[str]:
    """
    获取预训练权重路径，不存在则自动下载

    Args:
        model_type: 模型类型，如 'yolov5s.pt'
        version: YOLOv5 版本号
        weights_dir: 权重存放目录，为 None 时使用默认缓存目录
        download_missing: 本地不存在时是否自动下载

    Returns:
        权重文件的绝对路径字符串，或 None（下载失败或跳过）
    """
    if weights_dir is None:
        weights_dir = WEIGHTS_CACHE_DIR

    weights_path = weights_dir / model_type

    if weights_path.exists():
        logger.info(f"预训练权重已存在: {weights_path}")
        return str(weights_path)

    if not download_missing:
        logger.warning("权重文件不存在且已设置为不自动下载，将从头开始训练")
        return None

    url = _find_download_url(model_type, version)
    if not url:
        logger.error(f"未找到 {model_type} 的下载链接")
        return None

    logger.info(f"开始下载预训练权重: {url}")
    try:
        _download_with_progress(url, weights_path, f"下载 {model_type}")
        logger.info(f"权重下载完成: {weights_path}")
        return str(weights_path)
    except Exception as e:
        logger.error(f"下载权重失败: {e}")
        if weights_path.exists():
            weights_path.unlink()
        return None


def ensure_ultralytics_weights(model_name: str, weights_dir: Path) -> Optional[str]:
    """
    获取 Ultralytics 体系预训练权重，下载到项目 weights/ 目录。

    利用 Ultralytics 内置的 attempt_download_asset 下载模型权重，
    使 Ultralytics 模型的权重与 YOLOv5 模型统一存放在项目 weights/ 目录下，
    便于统一管理和离线复用。

    Args:
        model_name: 模型文件名，如 'yolov8n.pt' / 'yolo11m.pt'
        weights_dir: 权重存放目录，如 cfg.PATHS['weights_dir']

    Returns:
        权重文件的绝对路径字符串，或 None（下载失败）
    """
    from ultralytics.utils.downloads import attempt_download_asset

    target = weights_dir / model_name
    if target.exists():
        logger.info(f"预训练权重已存在: {target}")
        return str(target)

    logger.info(f"正在下载 Ultralytics 模型: {model_name}")
    try:
        result = attempt_download_asset(str(target))
        if Path(result).exists():
            logger.info(f"权重下载完成: {result}")
            return result
        logger.error(f"下载后文件未找到: {result}")
        return None
    except Exception as e:
        logger.error(f"下载 Ultralytics 权重失败: {e}")
        return None
