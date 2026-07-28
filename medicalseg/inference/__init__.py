"""
医学影像分割推理模块

提供模型推理和预测功能，主要导出：
- Predictor: 推理器类，封装模型加载、预处理、推理、结果保存等完整流程
"""

from .predictor import Predictor

__all__ = [
    'Predictor',
]
