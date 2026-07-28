"""
医学图像分割模型 - 模型工厂
根据配置对象创建相应的模型实例，统一模型创建入口。
"""
from typing import Any

from .unet import UNet
from .unetpp import UNetPlusPlus


def build_model(cfg: Any) -> Any:
    """
    根据配置对象构建分割模型
    
    支持的模型：
    - 'unet': 经典U-Net网络
    - 'unetpp': UNet++网络（嵌套U-Net，支持深度监督）
    
    Args:
        cfg: Config配置对象，需要包含以下属性：
             - cfg.model.name: 模型名称
             - cfg.data.in_channels: 输入图像通道数
             - cfg.data.num_classes: 分割类别数
             - cfg.model.bilinear: 是否使用双线性插值上采样
             - cfg.model.deep_supervision: 是否使用深度监督（仅UNet++支持）
    
    Returns:
        model: 构建好的模型实例（nn.Module子类）
    
    Raises:
        ValueError: 当指定的模型名称不存在时抛出
    """
    # 从配置中获取模型名称，转为小写方便匹配
    model_name = cfg.model.name.lower()
    
    # 根据模型名称创建对应的模型实例
    if model_name == 'unet':
        # 创建经典U-Net模型
        model = UNet(
            in_channels=cfg.data.in_channels,
            num_classes=cfg.data.num_classes,
            bilinear=cfg.model.bilinear
        )
    elif model_name == 'unetpp':
        # 创建UNet++模型，支持深度监督
        model = UNetPlusPlus(
            in_channels=cfg.data.in_channels,
            num_classes=cfg.data.num_classes,
            bilinear=cfg.model.bilinear,
            deep_supervision=cfg.model.deep_supervision
        )
    else:
        # 未知模型名称，抛出异常并提示支持的模型
        supported_models = ['unet', 'unetpp']
        raise ValueError(
            f"未知的模型名称: '{cfg.model.name}'。"
            f"当前支持的模型: {supported_models}"
        )
    
    return model
