"""
医学图像分割模型模块
包含各种分割网络结构和模型构建工厂。

导出内容：
- build_model: 模型工厂函数，根据配置创建模型
- UNet: 经典U-Net网络结构
- UNetPlusPlus: UNet++（嵌套U-Net）网络结构
- layers: 基础层模块（DoubleConv, Down, Up, OutConv）
"""
from .layers import DoubleConv, Down, Up, OutConv
from .unet import UNet
from .unetpp import UNetPlusPlus
from .model_factory import build_model

__all__ = [
    'build_model',
    'UNet',
    'UNetPlusPlus',
    'DoubleConv',
    'Down',
    'Up',
    'OutConv',
]
