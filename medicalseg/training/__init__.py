"""
医学图像分割训练模块
包含损失函数、评估指标、优化器、学习率调度器、早停机制和训练器。

导出内容：
损失函数（nn.Module 子类）：
- DiceLoss: Dice 损失函数
- FocalLoss: Focal 损失函数
- DiceFocalLoss: Dice + Focal 组合损失函数

评估指标（普通函数）：
- dice_coeff: Dice 系数
- iou_score: IoU 交并比
- precision: 精确率
- recall: 召回率
- accuracy: 准确率
- get_metrics: 一次性计算所有指标的辅助函数

优化器与学习率调度器：
- build_optimizer: 根据配置构建优化器（支持 AdamW/Adam/SGD）
- build_scheduler: 根据配置构建学习率调度器（支持 Cosine/Step/Plateau）

早停机制：
- EarlyStopping: 早停类，监控验证指标防止过拟合，自动保存最佳模型

训练器：
- Trainer: 训练器核心类，封装完整训练和验证流程
"""
from .losses import DiceLoss, FocalLoss, DiceFocalLoss
from .metrics import dice_coeff, iou_score, precision, recall, accuracy, get_metrics
from .optimizer import build_optimizer, build_scheduler
from .early_stopping import EarlyStopping
from .trainer import Trainer

__all__ = [
    'DiceLoss',
    'FocalLoss',
    'DiceFocalLoss',
    'dice_coeff',
    'iou_score',
    'precision',
    'recall',
    'accuracy',
    'get_metrics',
    'build_optimizer',
    'build_scheduler',
    'EarlyStopping',
    'Trainer',
]
