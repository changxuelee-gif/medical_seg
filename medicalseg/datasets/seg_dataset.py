"""
医学影像分割数据集模块

实现SegmentationDataset类用于加载图像-掩码配对数据，
FileListSegDataset类用于加载预先划分的文件列表数据，
以及train_val_test_split函数用于数据集划分。
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from ..io import load_image
from .base_dataset import BaseDataset


class SegmentationDataset(BaseDataset):
    """
    医学影像分割数据集
    
    加载图像目录和对应掩码目录中的配对数据，支持数据预处理/增强transform。
    通过文件名（不含后缀）匹配图像和掩码对。
    
    目录结构示例：
        image_dir/
            case001.png
            case002.png
            ...
        mask_dir/
            case001.png
            case002.png
            ...
    
    Args:
        image_dir: 图像目录路径（Path对象）
        mask_dir: 掩码目录路径（Path对象）
        transform: 数据预处理/增强变换，Compose对象或None
        image_suffix: 图像文件后缀，默认'.png'
        mask_suffix: 掩码文件后缀，默认'.png'
        
    Raises:
        ValueError: 当图像和掩码数量不匹配，或某些图像找不到对应掩码时
        FileNotFoundError: 当目录不存在时
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: Optional[Callable] = None,
        image_suffix: str = '.png',
        mask_suffix: str = '.png',
    ) -> None:
        """
        初始化分割数据集
        
        扫描目录收集样本对，验证图像和掩码一一对应。
        
        Args:
            image_dir: 图像目录路径
            mask_dir: 掩码目录路径
            transform: 数据变换函数（Compose对象）
            image_suffix: 图像文件后缀
            mask_suffix: 掩码文件后缀
        """
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_suffix = image_suffix.lower()
        self.mask_suffix = mask_suffix.lower()

        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")

        self.samples: List[Tuple[Path, Path]] = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        """
        收集并匹配图像-掩码样本对
        
        扫描image_dir下所有指定后缀的文件，通过文件名（不含后缀）
        在mask_dir中查找对应的掩码文件。
        
        Returns:
            样本对列表，每个元素为(image_path, mask_path)元组，按文件名排序
            
        Raises:
            ValueError: 当图像和掩码数量不一致，或找不到对应掩码时
        """
        image_files = sorted(
            [f for f in self.image_dir.iterdir() 
             if f.is_file() and f.suffix.lower() == self.image_suffix]
        )

        mask_files = sorted(
            [f for f in self.mask_dir.iterdir() 
             if f.is_file() and f.suffix.lower() == self.mask_suffix]
        )

        if len(image_files) != len(mask_files):
            raise ValueError(
                f"图像数量({len(image_files)})与掩码数量({len(mask_files)})不一致！\n"
                f"图像目录: {self.image_dir}\n掩码目录: {self.mask_dir}"
            )

        samples = []
        for img_path in image_files:
            mask_path = self.mask_dir / (img_path.stem + self.mask_suffix)
            if not mask_path.exists():
                raise ValueError(
                    f"图像 {img_path.name} 找不到对应的掩码文件: {mask_path.name}"
                )
            samples.append((img_path, mask_path))

        return samples

    def __len__(self) -> int:
        """
        返回数据集样本总数
        
        Returns:
            样本数量
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        根据索引加载并返回单个样本
        
        处理流程：
        1. 根据索引获取图像和掩码路径
        2. 使用load_image读取图像和掩码
        3. 将掩码二值化（>0.5设为1，否则为0）
        4. 确保图像和掩码是二维(H, W)数组
        5. 应用transform（如果提供）
        6. 返回处理后的图像和掩码张量
        
        Args:
            idx: 样本索引
            
        Returns:
            (image_tensor, mask_tensor)元组，shape为(1, H, W)
            
        Raises:
            IndexError: 索引超出范围时抛出
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"索引{idx}超出范围，数据集大小为{len(self.samples)}")

        img_path, mask_path = self.samples[idx]

        image = load_image(img_path, as_gray=True)
        mask = load_image(mask_path, as_gray=True)

        if image.ndim == 3:
            image = image[:, :, 0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = (mask > 0.5).astype(np.uint8)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


class FileListSegDataset(Dataset):
    """
    文件列表分割数据集
    
    接受预先划分好的 (image_path, mask_path) 文件对列表，
    用于加载训练/验证/测试数据。支持数据预处理/增强 transform。
    
    此数据集类是为配合 train_val_test_split 函数使用而设计的，
    与 SegmentationDataset（从目录扫描）功能类似但数据源不同。
    """

    def __init__(
        self,
        file_list: List[Tuple[Path, Path]],
        transform: Optional[Callable] = None
    ):
        """
        初始化文件列表数据集
        
        Args:
            file_list: (image_path, mask_path) 元组的列表
            transform: 数据预处理/增强变换（Compose 对象）
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        """
        返回数据集样本总数
        
        Returns:
            样本数量
        """
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
        根据索引加载并返回单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (image_tensor, mask_tensor): 形状均为 (1, H, W) 的张量
        """
        if idx < 0 or idx >= len(self.file_list):
            raise IndexError(f"索引 {idx} 超出范围，数据集大小为 {len(self.file_list)}")

        img_path, mask_path = self.file_list[idx]

        image = load_image(img_path, as_gray=True)
        mask = load_image(mask_path, as_gray=True)

        if image.ndim == 3:
            image = image[:, :, 0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = (mask > 0.5).astype(np.uint8)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


def train_val_test_split(
    image_dir: Path,
    mask_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_suffix: str = '.png',
    mask_suffix: str = '.png',
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    将数据集按比例划分为训练集、验证集和测试集
    
    按文件名排序后随机划分，使用固定种子保证划分可复现。
    
    Args:
        image_dir: 图像目录路径
        mask_dir: 掩码目录路径
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.2
        test_ratio: 测试集比例，默认0.1
        seed: 随机种子，默认42，保证可复现
        image_suffix: 图像文件后缀，默认'.png'
        mask_suffix: 掩码文件后缀，默认'.png'
        
    Returns:
        (train_files, val_files, test_files)元组，每个元素是(image_path, mask_path)列表
        
    Raises:
        ValueError: 当三个比例之和不等于1，或图像/掩码数量不匹配时
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"train_ratio({train_ratio}) + val_ratio({val_ratio}) + "
            f"test_ratio({test_ratio}) = {train_ratio + val_ratio + test_ratio}，必须等于1.0"
        )

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

    image_suffix = image_suffix.lower()
    mask_suffix = mask_suffix.lower()

    image_files = sorted(
        [f for f in image_dir.iterdir()
         if f.is_file() and f.suffix.lower() == image_suffix]
    )

    mask_files = sorted(
        [f for f in mask_dir.iterdir()
         if f.is_file() and f.suffix.lower() == mask_suffix]
    )

    if len(image_files) != len(mask_files):
        raise ValueError(
            f"图像数量({len(image_files)})与掩码数量({len(mask_files)})不一致！"
        )

    all_pairs = []
    for img_path in image_files:
        mask_path = mask_dir / (img_path.stem + mask_suffix)
        if not mask_path.exists():
            raise ValueError(
                f"图像 {img_path.name} 找不到对应的掩码文件: {mask_path.name}"
            )
        all_pairs.append((img_path, mask_path))

    random.seed(seed)
    random.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = all_pairs[:n_train]
    val_files = all_pairs[n_train:n_train + n_val]
    test_files = all_pairs[n_train + n_val:]

    return train_files, val_files, test_files

