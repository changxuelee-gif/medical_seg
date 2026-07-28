"""
医学影像分割推理模块

提供 Predictor 类，封装模型加载、图像预处理、推理预测、结果保存等完整推理流程。
支持单文件推理和批量目录推理，可生成对比图、叠加图等可视化结果。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..datasets.transforms import get_val_transforms
from ..io import load_image
from ..models.model_factory import build_model
from ..utils.device import get_device
from ..visualization import visualize_comparison, visualize_overlay


class Predictor:
    """
    医学影像分割推理器
    
    封装完整的推理流程：
    1. 加载训练好的模型检查点
    2. 对输入图像进行与验证集一致的预处理
    3. 执行模型前向推理
    4. 后处理（sigmoid、阈值二值化）
    5. 将结果resize回原始图像尺寸
    6. 结果可视化与保存
    
    支持：
    - 单张图片推理（numpy数组或文件路径）
    - 批量目录推理
    - 结果可视化（三图对比、叠加图）
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        cfg=None
    ):
        """
        初始化推理器
        
        加载模型检查点、构建模型、准备预处理transform。
        
        Args:
            checkpoint_path: 模型检查点文件路径（.pth文件）
            device: 计算设备（torch.device），若为None则自动检测
            cfg: 配置对象，若为None则从checkpoint中加载
        """
        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = get_device(verbose=False)
        self.device = device

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型检查点文件不存在: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = checkpoint['config']

        model_state_dict = checkpoint['model_state_dict']

        self.model = build_model(self.cfg)

        try:
            self.model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            print(f"警告: strict=True加载失败，尝试strict=False: {e}")
            self.model.load_state_dict(model_state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.val_transforms = get_val_transforms(self.cfg)

        print(f"模型加载完成，使用设备: {self.device}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理
        
        将输入numpy数组处理为模型可接受的张量格式，流程与验证集一致：
        1. 值域归一化到[0,1]
        2. 创建dummy掩码（因为transforms需要成对输入）
        3. 应用验证集transforms（Resize、去噪、CLAHE、归一化、ToTensor）
        4. 增加batch维度，移到指定设备
        
        Args:
            image: 输入图像numpy数组，形状(H,W)，值域[0,1]或[0,255]
            
        Returns:
            预处理后的图像张量，形状(1,1,H,W)，已移到self.device
        """
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        dummy_mask = np.zeros_like(image, dtype=np.uint8)

        image_tensor, _ = self.val_transforms(image, dummy_mask)

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        threshold: float = 0.5,
        return_numpy: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行分割预测
        
        支持两种输入方式：
        1. 文件路径（str或Path）：自动读取图像
        2. numpy数组：直接使用
        
        处理流程：
        1. 加载/读取图像，保存原始尺寸
        2. 预处理（resize到模型输入尺寸）
        3. 模型前向推理（torch.no_grad()）
        4. 处理深度监督输出（取最后一个）
        5. sigmoid得到概率图
        6. 阈值二值化得到预测掩码
        7. 将概率图和掩码resize回原始图像尺寸
        
        Args:
            image: 输入图像，可以是文件路径或numpy数组(H,W)
            threshold: 二值化阈值，默认0.5，大于阈值为前景（1）
            return_numpy: 是否返回numpy数组，默认为True
            
        Returns:
            (original_image, pred_mask, prob_map)元组：
            - original_image: 原始图像numpy数组(H,W)，值域[0,1]
            - pred_mask: 预测掩码numpy数组(H,W)，值域{0,1}，uint8类型（与原图同尺寸）
            - prob_map: 概率图numpy数组(H,W)，值域[0,1]，float32类型（与原图同尺寸）
        """
        if isinstance(image, (str, Path)):
            original_image = load_image(image, as_gray=True)
        else:
            original_image = image.copy().astype(np.float32)
            if original_image.max() > 1.0:
                original_image = original_image / 255.0

        orig_h, orig_w = original_image.shape[:2]

        input_tensor = self.preprocess(original_image)

        with torch.no_grad():
            outputs = self.model(input_tensor)

            if isinstance(outputs, list):
                outputs = outputs[-1]

            probs = torch.sigmoid(outputs)
            pred_mask = (probs > threshold).float()

        if return_numpy:
            pred_mask_small = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
            prob_map_small = probs.squeeze().cpu().numpy().astype(np.float32)

            pred_mask = cv2.resize(
                pred_mask_small,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )
            prob_map = cv2.resize(
                prob_map_small,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            original_image = original_image.astype(np.float32)

        return original_image, pred_mask, prob_map

    def predict_file(
        self,
        image_path: Union[str, Path],
        save_dir: Optional[Union[str, Path]] = None,
        save_overlay: bool = True,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对单个图像文件进行推理并可选保存结果
        
        Args:
            image_path: 输入图像文件路径
            save_dir: 结果保存目录，若为None则不保存
            save_overlay: 是否保存叠加图，默认True
            threshold: 二值化阈值，默认0.5
            
        Returns:
            (original_image, pred_mask, prob_map)元组，同predict方法
        """
        image_path = Path(image_path)
        original_image, pred_mask, prob_map = self.predict(image_path, threshold=threshold)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            stem = image_path.stem

            comparison_path = save_dir / f"{stem}_comparison.png"
            visualize_comparison(
                image=original_image,
                gt_mask=np.zeros_like(pred_mask),
                pred_mask=pred_mask,
                save_path=comparison_path,
                titles=('原始影像', '(无真实掩码)', '预测病灶')
            )

            if save_overlay:
                overlay_path = save_dir / f"{stem}_overlay.png"
                visualize_overlay(
                    image=original_image,
                    pred_mask=pred_mask,
                    save_path=overlay_path
                )

            mask_path = save_dir / f"{stem}_pred.png"
            pred_mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
            pred_mask_pil.save(mask_path)

            prob_path = save_dir / f"{stem}_prob.png"
            prob_map_pil = Image.fromarray((prob_map * 255).astype(np.uint8), mode='L')
            prob_map_pil.save(prob_path)

        return original_image, pred_mask, prob_map

    def predict_dir(
        self,
        input_dir: Union[str, Path],
        save_dir: Union[str, Path],
        pattern: str = '*.png',
        save_overlay: bool = True,
        threshold: float = 0.5
    ) -> int:
        """
        批量推理目录下的所有图像
        
        遍历input_dir下所有匹配pattern的文件，逐个进行推理，
        显示tqdm进度条，结果保存到save_dir。
        
        Args:
            input_dir: 输入图像目录
            save_dir: 结果保存目录
            pattern: 文件匹配模式，默认'*.png'
            save_overlay: 是否保存叠加图，默认True
            threshold: 二值化阈值，默认0.5
            
        Returns:
            处理的图像数量
        """
        input_dir = Path(input_dir)
        save_dir = Path(save_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        image_files = sorted(input_dir.glob(pattern))

        if len(image_files) == 0:
            print(f"警告: 在 {input_dir} 下未找到匹配 {pattern} 的文件")
            return 0

        save_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for image_path in tqdm(image_files, desc='推理进度'):
            self.predict_file(
                image_path=image_path,
                save_dir=save_dir,
                save_overlay=save_overlay,
                threshold=threshold
            )
            count += 1

        print(f"批量推理完成，共处理 {count} 张图像，结果保存到: {save_dir}")
        return count
