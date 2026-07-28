"""
医学影像分割数据预处理与增强模块

提供一系列可调用的Transform类，用于图像和掩码的同步预处理与数据增强。
所有Transform类均实现__call__方法，接收(image, mask)两个numpy数组，返回处理后的(image, mask)。

注意事项：
- image: 单通道float32类型，像素值范围[0, 1]，shape为(H, W)
- mask: 单通道uint8类型，二值图像（0或1），shape为(H, W)
- 几何变换必须同步应用到image和mask
- image使用双线性插值，mask使用最近邻插值以保证标签不变
"""

import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch


class Compose:
    """
    组合多个Transform变换，按顺序依次调用
    
    将多个数据预处理/增强操作组合成一个处理流水线，
    前一个Transform的输出作为后一个Transform的输入。
    
    Args:
        transforms: Transform对象列表，按顺序执行
        
    Example:
        >>> transforms = Compose([
        ...     Resize(256),
        ...     Normalize(),
        ...     ToTensor()
        ... ])
        >>> img_tensor, mask_tensor = transforms(image, mask)
    """

    def __init__(self, transforms: List[object]) -> None:
        """
        初始化Compose组合器
        
        Args:
            transforms: Transform对象列表
        """
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        依次执行所有Transform变换
        
        Args:
            image: 输入图像数组，shape(H, W)，float32 [0,1]
            mask: 输入掩码数组，shape(H, W)，uint8 {0,1}
            
        Returns:
            处理后的(image, mask)元组，类型由最后一个Transform决定
        """
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class Resize:
    """
    将图像和掩码resize到目标尺寸
    
    使用cv2.resize进行缩放：
    - image使用双线性插值（cv2.INTER_LINEAR），保证图像平滑
    - mask使用最近邻插值（cv2.INTER_NEAREST），保证标签值不变
    
    Args:
        size: 目标尺寸，可以是整数（正方形）或元组(height, width)
        
    Example:
        >>> resize = Resize(256)  # resize到256x256
        >>> img_resized, mask_resized = resize(image, mask)
    """

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        """
        初始化Resize变换
        
        Args:
            size: 目标尺寸，int表示正方形边长，(h, w)表示高和宽
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (size[0], size[1])

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行resize操作
        
        Args:
            image: 输入图像，shape(H, W)，float32 [0,1]
            mask: 输入掩码，shape(H, W)，uint8 {0,1}
            
        Returns:
            缩放后的(image, mask)，shape为(size_h, size_w)
        """
        h, w = self.size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)
        return image, mask


class Normalize:
    """
    将图像像素值归一化到[min_val, max_val]范围
    
    掩码保持二值0/1不变。如果图像所有像素值相同，则直接填充为min_val。
    
    Args:
        min_val: 归一化后的最小值，默认0.0
        max_val: 归一化后的最大值，默认1.0
        
    Note:
        由于io.load_image已经将图像归一化到[0,1]，
        此Transform主要用于需要其他归一化范围的场景（如[-1, 1]）
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """
        初始化Normalize变换
        
        Args:
            min_val: 目标最小值
            max_val: 目标最大值
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行归一化操作
        
        Args:
            image: 输入图像，shape(H, W)，float32
            mask: 输入掩码，shape(H, W)，uint8 {0,1}
            
        Returns:
            归一化后的(image, mask)
        """
        image = image.astype(np.float32)
        img_min = np.min(image)
        img_max = np.max(image)

        if img_max == img_min:
            image = np.full_like(image, self.min_val, dtype=np.float32)
        else:
            image = (image - img_min) / (img_max - img_min)
            image = image * (self.max_val - self.min_val) + self.min_val

        return image.astype(np.float32), mask


class ClaheEqualize:
    """
    对图像应用CLAHE（限制对比度自适应直方图均衡化）
    
    CLAHE用于医学影像的灰度矫正，增强局部对比度，
    比普通直方图均衡化更能抑制噪声放大。
    掩码保持不变。
    
    Args:
        clip_limit: 对比度限制阈值，默认2.0，值越大增强越强
        grid_size: 网格大小，默认(8, 8)，将图像划分为小块进行局部均衡
        
    Note:
        CLAHE需要uint8类型输入[0,255]，因此内部会先转换再转回float32 [0,1]
    """

    def __init__(self, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> None:
        """
        初始化CLAHE变换
        
        Args:
            clip_limit: 对比度限制参数
            grid_size: 局部均衡的网格大小
        """
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行CLAHE直方图均衡化
        
        Args:
            image: 输入图像，shape(H, W)，float32 [0,1]
            mask: 输入掩码，shape(H, W)，uint8 {0,1}
            
        Returns:
            增强后的(image, mask)，image为float32 [0,1]
        """
        image_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        image_eq = clahe.apply(image_uint8)
        image = image_eq.astype(np.float32) / 255.0
        return image, mask


class GaussianDenoise:
    """
    对图像应用高斯滤波去噪
    
    使用高斯模糊抑制图像噪声，掩码保持不变。
    
    Args:
        sigma: 高斯核标准差，默认0.5，值越大去噪效果越强但图像越模糊
        
    Note:
        核大小根据sigma自动计算：ksize = 2 * ceil(3 * sigma) + 1
    """

    def __init__(self, sigma: float = 0.5) -> None:
        """
        初始化高斯去噪变换
        
        Args:
            sigma: 高斯核标准差
        """
        self.sigma = sigma

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行高斯滤波去噪
        
        Args:
            image: 输入图像，shape(H, W)，float32 [0,1]
            mask: 输入掩码，shape(H, W)，uint8 {0,1}
            
        Returns:
            去噪后的(image, mask)
        """
        ksize = int(2 * np.ceil(3 * self.sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=self.sigma, sigmaY=self.sigma)
        return image.astype(np.float32), mask


class RandomHorizontalFlip:
    """
    随机水平翻转图像和掩码
    
    以概率p对图像和掩码进行同步水平翻转（左右翻转）。
    
    Args:
        p: 翻转概率，默认0.5
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        初始化随机水平翻转变换
        
        Args:
            p: 执行翻转的概率，取值范围[0, 1]
        """
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机执行水平翻转
        
        Args:
            image: 输入图像，shape(H, W)
            mask: 输入掩码，shape(H, W)
            
        Returns:
            翻转后的(image, mask)（若随机数小于p），否则原样返回
        """
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask


class RandomVerticalFlip:
    """
    随机垂直翻转图像和掩码
    
    以概率p对图像和掩码进行同步垂直翻转（上下翻转）。
    
    Args:
        p: 翻转概率，默认0.5
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        初始化随机垂直翻转变换
        
        Args:
            p: 执行翻转的概率，取值范围[0, 1]
        """
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机执行垂直翻转
        
        Args:
            image: 输入图像，shape(H, W)
            mask: 输入掩码，shape(H, W)
            
        Returns:
            翻转后的(image, mask)（若随机数小于p），否则原样返回
        """
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask


class RandomRotation:
    """
    随机旋转图像和掩码
    
    在[-degrees, +degrees]范围内随机选择一个角度进行旋转，
    image使用双线性插值，mask使用最近邻插值。
    旋转后图像边界使用0填充。
    
    Args:
        degrees: 旋转角度范围，默认15度，表示在[-15°, 15°]之间随机旋转
        
    Note:
        旋转中心为图像中心
    """

    def __init__(self, degrees: float = 15) -> None:
        """
        初始化随机旋转变换
        
        Args:
            degrees: 最大旋转角度（正数），实际旋转角度为[-degrees, degrees]
        """
        self.degrees = degrees

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行随机旋转
        
        Args:
            image: 输入图像，shape(H, W)
            mask: 输入掩码，shape(H, W)
            
        Returns:
            旋转后的(image, mask)，保持原尺寸
        """
        angle = random.uniform(-self.degrees, self.degrees)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = (mask > 0.5).astype(np.uint8)
        return image, mask


class ToTensor:
    """
    将numpy数组转换为PyTorch Tensor
    
    处理流程：
    1. 确保输入是二维(H, W)数组（灰度图）
    2. 增加通道维度，变为(1, H, W)
    3. 转换为torch.Tensor，float32类型
    
    注意：image和mask都会增加通道维度，mask转为float32以匹配模型输入
    
    Example:
        >>> to_tensor = ToTensor()
        >>> img_t, mask_t = to_tensor(image, mask)
        >>> img_t.shape  # torch.Size([1, 256, 256])
        >>> mask_t.shape  # torch.Size([1, 256, 256])
    """

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将numpy数组转换为Tensor
        
        Args:
            image: 输入图像，shape(H, W)，float32 [0,1]
            mask: 输入掩码，shape(H, W)，uint8 {0,1}
            
        Returns:
            (image_tensor, mask_tensor)，shape均为(1, H, W)，dtype=torch.float32
        """
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
            image = image[np.newaxis, :, :]

        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
            mask = mask[np.newaxis, :, :]

        image_tensor = torch.from_numpy(image.astype(np.float32)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).float()
        return image_tensor, mask_tensor


def get_train_transforms(cfg) -> Compose:
    """
    根据配置构建训练集的数据预处理与增强流水线
    
    训练集transform顺序：
    1. Resize：缩放到配置的目标尺寸
    2. GaussianDenoise：高斯去噪
    3. ClaheEqualize：CLAHE直方图均衡化
    4. Normalize：归一化（如果配置启用）
    5. RandomHorizontalFlip：随机水平翻转（如果启用增强）
    6. RandomVerticalFlip：随机垂直翻转（如果启用增强）
    7. RandomRotation：随机旋转（如果启用增强）
    8. ToTensor：转为PyTorch Tensor
    
    Args:
        cfg: Config配置对象，需要包含：
            - data.img_size: 目标图像尺寸
            - data.normalize: 是否归一化
            - data.augment: 是否启用数据增强
            
    Returns:
        Compose对象，包含所有训练用transform
    """
    transforms_list = []

    transforms_list.append(Resize(cfg.data.img_size))
    transforms_list.append(GaussianDenoise(sigma=0.5))
    transforms_list.append(ClaheEqualize(clip_limit=2.0, grid_size=(8, 8)))

    if cfg.data.get('normalize', True):
        transforms_list.append(Normalize(min_val=0.0, max_val=1.0))

    if cfg.data.get('augment', True):
        transforms_list.append(RandomHorizontalFlip(p=0.5))
        transforms_list.append(RandomVerticalFlip(p=0.5))
        transforms_list.append(RandomRotation(degrees=15))

    transforms_list.append(ToTensor())

    return Compose(transforms_list)


def get_val_transforms(cfg) -> Compose:
    """
    根据配置构建验证集/测试集的数据预处理流水线
    
    验证集不使用数据增强，只进行必要的预处理：
    1. Resize：缩放到配置的目标尺寸
    2. GaussianDenoise：高斯去噪
    3. ClaheEqualize：CLAHE直方图均衡化
    4. Normalize：归一化（如果配置启用）
    5. ToTensor：转为PyTorch Tensor
    
    Args:
        cfg: Config配置对象，需要包含：
            - data.img_size: 目标图像尺寸
            - data.normalize: 是否归一化
            
    Returns:
        Compose对象，包含所有验证用transform（无数据增强）
    """
    transforms_list = []

    transforms_list.append(Resize(cfg.data.img_size))
    transforms_list.append(GaussianDenoise(sigma=0.5))
    transforms_list.append(ClaheEqualize(clip_limit=2.0, grid_size=(8, 8)))

    if cfg.data.get('normalize', True):
        transforms_list.append(Normalize(min_val=0.0, max_val=1.0))

    transforms_list.append(ToTensor())

    return Compose(transforms_list)

