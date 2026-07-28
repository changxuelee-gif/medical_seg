"""
医学图像分割可视化模块

提供训练曲线绘制和分割结果可视化功能：

训练曲线绘图（plotter）：
- setup_chinese_font: 配置 matplotlib 中文字体，解决中文显示乱码
- plot_single_curve: 绘制单个指标曲线（支持 loss/dice/iou 等）
- plot_metric_curves: 批量绘制 loss、dice、iou 三条主要训练曲线

分割结果可视化（visualizer）：
- visualize_comparison: 原始影像/真实掩码/预测掩码三栏对比图
- visualize_overlay: 在原始影像上彩色叠加显示分割病灶
- save_prediction_grid: 批量预测结果网格布局展示
"""

from .plotter import setup_chinese_font, plot_single_curve, plot_metric_curves
from .visualizer import visualize_comparison, visualize_overlay, save_prediction_grid

__all__ = [
    # plotter 训练曲线绘图
    'setup_chinese_font',
    'plot_single_curve',
    'plot_metric_curves',
    # visualizer 分割结果可视化
    'visualize_comparison',
    'visualize_overlay',
    'save_prediction_grid',
]
