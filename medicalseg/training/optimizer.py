"""
医学图像分割训练 - 优化器与学习率调度器模块
本模块提供优化器和学习率调度器的构建函数，支持多种常用配置。
"""
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Union


def build_optimizer(model: torch.nn.Module, cfg) -> Optimizer:
    """
    根据配置构建优化器。

    参数说明：
        model: 需要训练的 PyTorch 模型，其可训练参数将被传入优化器
        cfg: 配置对象，需包含以下属性：
            - cfg.training.optimizer (str): 优化器类型，可选值：'adamw'（默认）、'adam'、'sgd'
            - cfg.training.lr (float): 初始学习率
            - cfg.training.weight_decay (float): 权重衰减系数（L2正则化）

    返回：
        optimizer: 构建好的优化器实例

    支持的优化器：
        - adamw: AdamW 优化器（推荐，解耦权重衰减）
        - adam: 标准 Adam 优化器
        - sgd: 带动量的随机梯度下降优化器（预留选项）
    """
    # 获取优化器类型，默认为 adamw
    optimizer_type = getattr(cfg.training, 'optimizer', 'adamw').lower()
    # 获取学习率，默认 1e-3
    lr = getattr(cfg.training, 'lr', 1e-3)
    # 获取权重衰减，默认 1e-4
    weight_decay = getattr(cfg.training, 'weight_decay', 1e-4)

    if optimizer_type == 'adamw':
        # AdamW 优化器：对权重衰减进行解耦，通常比 Adam 泛化性能更好
        # 参数说明：
        #   params: 模型需要更新的参数
        #   lr: 初始学习率
        #   weight_decay: L2 正则化系数（权重衰减）
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        # 标准 Adam 优化器
        # 参数说明：
        #   params: 模型需要更新的参数
        #   lr: 初始学习率
        #   weight_decay: L2 正则化系数
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        # 带动量的 SGD 优化器（预留选项）
        # 参数说明：
        #   params: 模型需要更新的参数
        #   lr: 初始学习率
        #   momentum: 动量系数，默认 0.9
        #   weight_decay: L2 正则化系数
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        # 未知优化器类型，默认回退到 AdamW
        print(f"[警告] 未知优化器类型 '{optimizer_type}'，将使用默认优化器 AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    cfg,
    num_epochs: Optional[int] = None
) -> Optional[Union[_LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]]:
    """
    根据配置构建学习率调度器。

    参数说明：
        optimizer: 已经构建好的优化器实例
        cfg: 配置对象，需包含以下属性：
            - cfg.training.scheduler (str): 调度器类型，可选值：'cosine'（默认）、'step'、'plateau'、'none'
        num_epochs (int, 可选): 训练总轮数，仅 cosine 调度器需要此参数设置 T_max

    返回：
        scheduler: 构建好的学习率调度器实例，如果配置为 'none' 或未知类型则返回 None

    支持的调度器：
        - cosine: 余弦退火学习率调度器（推荐）
          学习率按照余弦函数从初始值衰减到 eta_min
        - step: 步进式学习率调度器
          每固定轮数（step_size=10）将学习率乘以 gamma=0.5（衰减一半）
        - plateau: 基于验证指标的自适应调度器
          当监控的指标在 patience=5 轮内没有提升时，将学习率乘以 factor=0.5
        - none/None: 不使用学习率调度器，返回 None

    注意：
        对于 'plateau' 调度器，在训练循环中需要调用 scheduler.step(metric)，
        传入当前验证集的监控指标值（如 Dice 系数或 loss）。
        其他调度器在每轮结束后调用 scheduler.step() 即可。
    """
    # 获取调度器类型，默认为 cosine
    scheduler_type = getattr(cfg.training, 'scheduler', 'cosine').lower()

    if scheduler_type == 'cosine':
        # 余弦退火学习率调度器
        # 参数说明：
        #   optimizer: 绑定的优化器
        #   T_max: 余弦周期的一半（通常设为总训练轮数）
        #   eta_min: 最小学习率，衰减不会低于此值，默认 1e-6
        if num_epochs is None:
            print("[警告] CosineAnnealingLR 需要 num_epochs 参数，将默认设置为 100")
            t_max = 100
        else:
            t_max = num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=1e-6
        )
    elif scheduler_type == 'step':
        # 步进式学习率调度器
        # 参数说明：
        #   optimizer: 绑定的优化器
        #   step_size: 每多少轮衰减一次学习率，默认 10 轮
        #   gamma: 学习率衰减系数，默认 0.5（每次衰减为原来的一半）
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    elif scheduler_type == 'plateau':
        # 基于验证指标的自适应学习率调度器
        # 参数说明：
        #   optimizer: 绑定的优化器
        #   mode: 'max' 表示监控指标越大越好（如 Dice、IoU），'min' 表示越小越好（如 loss）
        #   factor: 学习率衰减系数，默认 0.5
        #   patience: 容忍多少轮没有提升才衰减学习率，默认 5 轮
        #   verbose: 是否打印学习率变化信息，默认 True
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif scheduler_type in ['none', 'None', '']:
        # 不使用学习率调度器
        scheduler = None
    else:
        # 未知调度器类型，返回 None
        print(f"[警告] 未知学习率调度器类型 '{scheduler_type}'，将不使用调度器")
        scheduler = None

    return scheduler
