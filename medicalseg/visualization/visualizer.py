"""
医学图像分割结果可视化模块

提供分割预测结果的可视化功能，包括：
- 原始影像/真实掩码/预测掩码三栏对比
- 原始影像上叠加分割结果（彩色高亮显示病灶）
- 批量预测结果网格展示
"""

# 重要：在导入 pyplot 之前设置后端为 Agg，确保无 GUI 环境下也能保存图片
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """
    内部辅助函数：将图像归一化到 [0, 1] 范围
    
    自动处理不同值域的输入：
    - 如果像素最大值 > 1，认为是 [0, 255] 范围，除以 255 归一化
    - 否则认为已经是 [0, 1] 范围，直接返回
    
    Args:
        img: 输入图像 numpy 数组
        
    Returns:
        归一化到 [0, 1] 的图像数组
    """
    if img.max() > 1.0:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def visualize_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Optional[Path] = None,
    titles: Optional[Tuple[str, str, str]] = None
) -> Optional[plt.Figure]:
    """
    可视化原始影像、真实掩码、预测掩码的三栏对比图
    
    创建 1x3 的子图布局，分别展示：
    1. 原始医学影像（灰度）
    2. 医生标注的真实病灶掩码
    3. 模型预测的病灶掩码
    
    便于直观对比模型预测结果与真实标注的差异。
    
    Args:
        image: 原始影像 numpy 数组，形状 (H, W)，值域 [0, 1] 或 [0, 255]
        gt_mask: 真实掩码（Ground Truth）numpy 数组，形状 (H, W)，二值 0/1
        pred_mask: 预测掩码 numpy 数组，形状 (H, W)，二值 0/1
        save_path: 图片保存路径（Path 对象），若为 None 则不保存，返回 Figure 对象
        titles: 三个子图的标题元组，若为 None 则使用默认中文标题
        
    Returns:
        如果 save_path 为 None，返回 matplotlib Figure 对象；否则返回 None
    """
    # 归一化图像到 [0, 1]
    image = _normalize_image(image)

    # 确保掩码是二值的（>0.5 为前景）
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    pred_mask = (pred_mask > 0.5).astype(np.float32)

    # 使用默认标题或用户自定义标题
    if titles is None:
        titles = ('原始影像', '真实病灶', '预测病灶')

    # 创建 1x3 子图，设置合适的尺寸
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ========== 绘制原始影像 ==========
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')  # 隐藏坐标轴

    # ========== 绘制真实掩码 ==========
    axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    # ========== 绘制预测掩码 ==========
    axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(titles[2], fontsize=14, fontweight='bold', pad=10)
    axes[2].axis('off')

    # 调整子图间距
    plt.tight_layout(pad=2.0)

    # 保存或返回 Figure
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def visualize_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Optional[Path] = None,
    alpha: float = 0.5,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> Optional[plt.Figure]:
    """
    在原始影像上以彩色叠加方式显示分割结果
    
    将灰度影像转换为 RGB，在预测为病灶的区域用指定颜色高亮显示，
    可以直观地看到病灶在原始影像中的位置和形状。
    
    叠加原理：
    - 背景区域保持原始灰度值
    - 病灶区域：原始灰度 * (1-alpha) + 指定颜色 * alpha
    - alpha 越大，颜色叠加越明显
    
    Args:
        image: 原始影像 numpy 数组，形状 (H, W)，值域 [0, 1] 或 [0, 255]
        pred_mask: 预测掩码 numpy 数组，形状 (H, W)，二值 0/1 或概率值
        save_path: 图片保存路径，若为 None 则返回 Figure 对象
        alpha: 颜色叠加透明度，范围 [0, 1]，默认 0.5
               - 0: 完全透明，只显示原始影像
               - 1: 完全不透明，病灶区域显示为纯色
        color: 叠加颜色 RGB 元组，每个通道范围 [0, 1]，默认红色 (1.0, 0.0, 0.0)
               其他常用颜色：
               - 红色：(1.0, 0.0, 0.0)
               - 绿色：(0.0, 1.0, 0.0)
               - 蓝色：(0.0, 0.0, 1.0)
               - 黄色：(1.0, 1.0, 0.0)
        
    Returns:
        如果 save_path 为 None，返回 matplotlib Figure 对象；否则返回 None
    """
    # 归一化图像到 [0, 1]
    image = _normalize_image(image)

    # 将灰度图转换为 RGB 三通道（复制灰度值到 R、G、B）
    image_rgb = np.stack([image, image, image], axis=-1)

    # 确保掩码在 [0, 1] 范围（支持概率图输入）
    pred_mask = pred_mask.astype(np.float32)
    if pred_mask.max() > 1.0:
        pred_mask = pred_mask / 255.0

    # 创建叠加颜色数组，形状与 image_rgb 相同 (H, W, 3)
    color_array = np.zeros_like(image_rgb)
    color_array[:, :, 0] = color[0]
    color_array[:, :, 1] = color[1]
    color_array[:, :, 2] = color[2]

    # 计算叠加结果：使用 alpha 混合
    # 公式：result = background * (1 - alpha * mask) + color * (alpha * mask)
    # 其中 mask 是 0/1 二值图，这样背景区域不变，病灶区域被染色
    mask_3d = np.stack([pred_mask > 0.5] * 3, axis=-1)
    overlay = image_rgb.copy()
    overlay[mask_3d] = image_rgb[mask_3d] * (1 - alpha) + color_array[mask_3d] * alpha

    # 创建图形并显示
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.clip(overlay, 0, 1))  # clip 确保像素值在 [0, 1] 范围内
    ax.set_title('分割结果叠加图', fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')

    plt.tight_layout()

    # 保存或返回 Figure
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def save_prediction_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    preds: List[np.ndarray],
    save_path: Path,
    nrow: int = 4,
    figsize: Tuple[int, int] = (15, 15)
) -> None:
    """
    批量保存预测结果为网格布局图片
    
    将多张影像的分割叠加结果排列在网格中，便于快速批量检查模型预测效果。
    每个格子显示一张影像的 overlay 效果（原始影像 + 红色病灶叠加）。
    
    Args:
        images: 原始影像列表，每个元素形状 (H, W)
        masks: 真实掩码列表（暂未单独绘制，用于预留扩展），每个元素形状 (H, W)
        preds: 预测掩码列表，每个元素形状 (H, W)
        save_path: 图片保存路径（Path 对象）
        nrow: 网格每行显示的图片数量，默认 4 列
        figsize: 整个图形的尺寸（宽, 高），默认 (15, 15)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 计算需要的行数：总数 / 列数，向上取整
    num_samples = len(images)
    ncol = nrow
    nrows = int(np.ceil(num_samples / ncol))

    # 创建网格子图
    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)
    
    # 将 axes 转为一维数组，便于统一索引（处理单行/单列情况）
    if nrows == 1 and ncol == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 逐个绘制每张图片的叠加结果
    for i in range(num_samples):
        ax = axes[i]
        
        # 归一化图像
        img = _normalize_image(images[i])
        
        # 转换为 RGB 并叠加预测结果
        img_rgb = np.stack([img, img, img], axis=-1)
        pred = (preds[i] > 0.5).astype(np.float32)
        
        # 创建叠加颜色数组，与 img_rgb 形状相同 (H, W, 3)
        color_array = np.zeros_like(img_rgb)
        color_array[:, :, 0] = 1.0  # R
        color_array[:, :, 1] = 0.0  # G
        color_array[:, :, 2] = 0.0  # B
        
        # 构建 3D 掩码
        mask_3d = np.stack([pred > 0.5] * 3, axis=-1)
        
        # Alpha 混合：病灶区域显示红色叠加
        overlay = img_rgb.copy()
        overlay[mask_3d] = img_rgb[mask_3d] * 0.5 + color_array[mask_3d] * 0.5
        
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f'样本 #{i+1}', fontsize=10, fontweight='bold')
        ax.axis('off')

    # 隐藏多余的空白子图（当样本数不是网格整数倍时）
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    # 调整布局
    plt.tight_layout(pad=1.5)

    # 保存图片
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[visualizer] 预测结果网格已保存至: {save_path}")
