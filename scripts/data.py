"""
数据准备和路径管理模块
合并原 path_checker + data_preparer，全部改为纯函数
修复：test_ratio 被忽略、整数截断丢失数据、XML 坐标无效验证
"""

from __future__ import annotations

import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

logger = logging.getLogger('yolov5_trainer')


# ==================== 路径管理 ====================

def create_directories(paths: list[Path]) -> None:
    """创建多个目录（如已存在则跳过）"""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def check_data_exists(images_dir: Path, labels_dir: Path) -> bool:
    """检查训练数据（图片和标注）是否存在"""
    images_exist = any(images_dir.glob("*.*"))
    labels_exist = any(labels_dir.glob("*.*"))

    if not images_exist:
        logger.warning(f"未在 {images_dir} 中找到图片文件")
        return False
    if not labels_exist:
        logger.warning(f"未在 {labels_dir} 中找到标注文件")
        return False

    logger.info("数据文件检查通过")
    return True


# ==================== 数据准备 ====================

def _convert_single_xml(xml_file: Path, class_list: list[str]) -> None:
    """
    转换单个 XML 标注文件为 YOLO TXT 格式
    修复：验证坐标有效性（xmax > xmin, ymax > ymin）
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    txt_file = xml_file.with_suffix('.txt')

    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_list:
            logger.warning(f"{xml_file.name} 中存在未在 class_list 中的类别: {cls_name}，跳过")
            continue
        cls_id = class_list.index(cls_name)

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 修复：验证坐标有效性
        if xmax <= xmin or ymax <= ymin:
            logger.warning(
                f"{xml_file.name} 中 {cls_name} 坐标无效 "
                f"(xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})，跳过"
            )
            continue

        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def convert_xml_to_yolo(labels_dir: Path) -> list[str]:
    """将 VOC XML 批量转换为 YOLO TXT 格式，返回类别列表"""
    xml_files = list(labels_dir.glob("*.xml"))
    if not xml_files:
        logger.warning("未找到 XML 标注文件")
        return []

    # 提取所有类别
    classes: set[str] = set()
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            for obj in tree.getroot().findall('object'):
                name = obj.find('name')
                if name is not None and name.text:
                    classes.add(name.text)
        except Exception as e:
            logger.warning(f"解析 {xml_file.name} 失败: {e}")

    class_list = sorted(classes)
    logger.info(f"发现 {len(class_list)} 个类别: {class_list}")

    # 逐个转换
    converted = 0
    for xml_file in xml_files:
        try:
            _convert_single_xml(xml_file, class_list)
            converted += 1
        except Exception as e:
            logger.error(f"转换 {xml_file.name} 失败: {e}")

    logger.info(f"XML 转换完成: {converted}/{len(xml_files)}")
    return class_list


def write_classes_file(classes_file: Path, class_list: list[str]) -> None:
    """写入类别名称文件"""
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls_name in class_list:
            f.write(f"{cls_name}\n")
    logger.info(f"类别文件已保存: {classes_file} ({len(class_list)} 类)")


def generate_classes_from_labels(labels_dir: Path) -> list[str]:
    """从 YOLO TXT 标注文件中提取类别 ID，生成默认类别名"""
    ids: set[int] = set()
    for txt_file in labels_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        ids.add(int(parts[0]))
        except Exception as e:
            logger.warning(f"读取 {txt_file.name} 失败: {e}")

    if not ids:
        logger.warning("未从标注中发现类别 ID，使用默认类别 'class0'")
        return ['class0']

    return [f'class{i}' for i in sorted(ids)]


def split_dataset(
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, int]:
    """
    划分数据集（训练/验证/测试）

    修复：
    1. 支持 test_ratio（原代码完全忽略了它）
    2. 使用 round + 补足方式，避免整数截断导致数据丢失
    """
    # 收集所有图片
    image_files: list[Path] = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_dir.glob(ext))

    if not image_files:
        raise FileNotFoundError(f"在 {images_dir} 中未找到图片文件")

    # 随机打乱
    random.seed(seed)
    random.shuffle(image_files)

    total = len(image_files)

    # 比例归一化
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError(f"数据集比例之和必须 > 0，当前: {total_ratio}")
    tr, vr, trr = train_ratio / total_ratio, val_ratio / total_ratio, test_ratio / total_ratio

    # 使用 round + 补足，解决整数截断丢失数据的问题
    n_train = round(total * tr)
    n_val = round(total * vr)
    n_test = total - n_train - n_val  # 余量全部归 test，确保不丢失

    # 如果 test_ratio 为 0（或接近 0），test 合并到 val
    if test_ratio < 1e-9:
        n_val += n_test
        n_test = 0

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:] if n_test > 0 else []

    # 保存
    def _save(filename: str, files: list[Path]):
        path = output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            for img in files:
                f.write(f"{img}\n")

    _save('train.txt', train_files)
    _save('val.txt', val_files)
    if test_files:
        _save('test.txt', test_files)

    result: dict[str, int] = {'train': len(train_files), 'val': len(val_files)}
    if test_files:
        result['test'] = len(test_files)

    logger.info(
        f"数据集划分: 训练 {result['train']} 张, "
        f"验证 {result['val']} 张"
        + (f", 测试 {result['test']} 张" if 'test' in result else "")
    )
    return result


def create_data_yaml(project_dir: Path, classes_file: Path) -> Path:
    """创建 YOLO data.yaml，返回文件路径"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    data = {
        'path': str(project_dir.parent),
        'train': str(project_dir / 'train.txt'),
        'val': str(project_dir / 'val.txt'),
        'test': '',
        'nc': len(classes),
        'names': classes,
    }
    yaml_file = project_dir / 'data.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"数据配置文件已创建: {yaml_file}")
    return yaml_file


def prepare_data(
    images_dir: Path,
    labels_dir: Path,
    classes_file: Path,
    project_dir: Path,
    annotation_format: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    random_seed: int = 42,
) -> None:
    """完整的训练数据准备流程"""
    logger.info("开始准备训练数据...")

    if not check_data_exists(images_dir, labels_dir):
        raise FileNotFoundError("训练数据不完整")

    # 处理标注格式
    if annotation_format == 'xml':
        classes = convert_xml_to_yolo(labels_dir)
        if classes:
            write_classes_file(classes_file, classes)
        else:
            logger.warning("XML 转换未产生类别，尝试从已有 TXT 标注生成")
    elif annotation_format == 'txt':
        logger.info("使用 YOLO TXT 格式标注，无需转换")
    else:
        raise ValueError(f"不支持的标注格式: {annotation_format}")

    # 生成类别文件（如果还不存在）
    if not classes_file.exists():
        class_list = generate_classes_from_labels(labels_dir)
        write_classes_file(classes_file, class_list)
    else:
        logger.info(f"类别文件已存在: {classes_file}")

    # 划分数据集
    split_dataset(
        images_dir, project_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=random_seed,
    )

    # 创建 data.yaml
    create_data_yaml(project_dir, classes_file)

    logger.info("数据准备完成")
