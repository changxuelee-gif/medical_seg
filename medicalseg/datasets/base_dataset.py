"""
数据集抽象基类模块

定义所有数据集的抽象基类BaseDataset，继承自torch.utils.data.Dataset，
要求子类必须实现__len__和__getitem__方法。
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    数据集抽象基类
    
    所有自定义数据集都应继承此类，并实现以下两个抽象方法：
    - __len__: 返回数据集样本总数
    - __getitem__: 根据索引返回单个样本（图像和标签）
    
    继承torch.utils.data.Dataset以支持PyTorch DataLoader的并行加载。
    
    Example:
        >>> class MyDataset(BaseDataset):
        ...     def __init__(self, ...):
        ...         super().__init__()
        ...         # 初始化逻辑
        ...     
        ...     def __len__(self):
        ...         return len(self.samples)
        ...     
        ...     def __getitem__(self, idx):
        ...         # 加载并返回样本
        ...         return image, label
    """

    def __init__(self) -> None:
        """
        初始化基类
        """
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        """
        返回数据集的样本总数
        
        Returns:
            数据集中的样本数量（整数）
            
        Raises:
            NotImplementedError: 子类未实现此方法时抛出
        """
        raise NotImplementedError("子类必须实现__len__方法")

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        根据索引获取单个样本
        
        Args:
            idx: 样本索引，范围[0, len(self)-1]
            
        Returns:
            样本数据，通常为(image, label)元组
            
        Raises:
            NotImplementedError: 子类未实现此方法时抛出
            IndexError: 索引超出范围时抛出
        """
        raise NotImplementedError("子类必须实现__getitem__方法")

