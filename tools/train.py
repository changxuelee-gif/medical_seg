#!/usr/bin/env python3
"""
医学图像分割训练入口脚本

使用方法：
    python tools/train.py                           # 使用默认配置 configs/default.yaml 训练
    python tools/train.py --config configs/my.yaml  # 使用指定配置文件训练
    python tools/train.py --resume checkpoints/last_model.pth  # 从检查点恢复训练（预留功能）

功能说明：
    1. 解析命令行参数
    2. 设置 sys.path 确保能正确导入 medicalseg 包
    3. 加载配置文件
    4. 初始化日志
    5. 固定随机种子保证可复现性
    6. 自动检测并设置计算设备（GPU/CPU）
    7. 构建分割模型
    8. 划分训练/验证/测试集
    9. 创建数据增强/预处理变换
    10. 构建数据集和 DataLoader
    11. 构建损失函数、优化器、学习率调度器、早停机制
    12. 创建 Trainer 实例并开始训练
    13. 训练完成保存模型和指标
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ========== 设置项目根目录到 sys.path ==========
# 当前脚本位于 tools/train.py，向上一级即为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 现在可以导入 medicalseg 包
from medicalseg.utils.config import load_config, get_default_config_path
from medicalseg.utils.logger import setup_logger
from medicalseg.utils.seed import set_seed
from medicalseg.utils.device import get_device
from medicalseg.models.model_factory import build_model
from medicalseg.datasets import (
    get_train_transforms,
    get_val_transforms,
    train_val_test_split,
    FileListSegDataset,
)
from medicalseg.io import load_image
from medicalseg.training import (
    DiceFocalLoss,
    build_optimizer,
    build_scheduler,
    EarlyStopping,
    Trainer,
)


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='医学图像分割训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python tools/train.py
  python tools/train.py --config configs/default.yaml
  python tools/train.py --resume checkpoints/best_model.pth
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（YAML 格式），默认为 configs/default.yaml'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='断点续训的检查点路径（预留功能，暂未完整实现）'
    )
    return parser.parse_args()


def main():
    """训练主函数"""
    # ========== 1. 解析命令行参数 ==========
    args = parse_args()

    # ========== 2. 加载配置文件 ==========
    # 如果未指定配置文件，使用默认配置路径
    if args.config is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(args.config).resolve()

    print(f"加载配置文件: {config_path}")
    cfg = load_config(config_path=config_path, default_config_path=get_default_config_path())

    # ========== 3. 初始化日志 ==========
    # 从配置获取日志目录，默认为项目根目录下的 logs/
    log_dir = getattr(cfg.paths, 'log_dir', 'logs')
    log_dir = PROJECT_ROOT / log_dir
    logger = setup_logger(log_dir=log_dir)

    logger.info("=" * 80)
    logger.info("医学图像分割训练启动")
    logger.info("=" * 80)

    # ========== 4. 固定随机种子 ==========
    seed = getattr(cfg, 'seed', 42)
    set_seed(seed)

    # ========== 5. 获取计算设备 ==========
    device = get_device(verbose=True)

    # ========== 6. 构建模型 ==========
    logger.info("正在构建模型...")
    model = build_model(cfg)
    model = model.to(device)

    # 打印模型参数量信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型: {cfg.model.name}")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数量: {trainable_params:,}")

    # ========== 7. 准备数据路径 ==========
    # 数据目录：优先从配置读取，默认为 data/raw/simulated
    data_dir = getattr(cfg.paths, 'data_dir', 'data/raw/simulated')
    data_dir = PROJECT_ROOT / data_dir
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'

    logger.info(f"数据目录: {data_dir}")
    logger.info(f"  图像目录: {image_dir}")
    logger.info(f"  掩码目录: {mask_dir}")

    # 检查目录是否存在
    if not image_dir.exists():
        logger.error(f"图像目录不存在: {image_dir}")
        logger.info("提示：请先运行数据生成脚本，例如: python tools/generate_sim_data.py")
        sys.exit(1)
    if not mask_dir.exists():
        logger.error(f"掩码目录不存在: {mask_dir}")
        sys.exit(1)

    # ========== 8. 划分训练/验证/测试集 ==========
    train_ratio = getattr(cfg.data, 'train_ratio', 0.7)
    val_ratio = getattr(cfg.data, 'val_ratio', 0.2)
    test_ratio = getattr(cfg.data, 'test_ratio', 0.1)

    logger.info("正在划分数据集...")
    logger.info(f"  划分比例 - 训练: {train_ratio}, 验证: {val_ratio}, 测试: {test_ratio}")

    train_files, val_files, test_files = train_val_test_split(
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    logger.info(f"数据集划分完成:")
    logger.info(f"  训练集: {len(train_files)} 个样本")
    logger.info(f"  验证集: {len(val_files)} 个样本")
    logger.info(f"  测试集: {len(test_files)} 个样本")

    # ========== 9. 创建数据变换 ==========
    logger.info("正在创建数据预处理/增强变换...")
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # ========== 10. 创建数据集 ==========
    logger.info("正在构建数据集...")
    # 训练集使用数据增强
    train_dataset = FileListSegDataset(
        file_list=train_files,
        transform=train_transform
    )
    # 验证集不使用数据增强
    val_dataset = FileListSegDataset(
        file_list=val_files,
        transform=val_transform
    )

    # ========== 11. 创建 DataLoader ==========
    batch_size = getattr(cfg.data, 'batch_size', 4)
    num_workers = getattr(cfg.data, 'num_workers', 2)

    logger.info("正在创建 DataLoader...")
    logger.info(f"  批大小 (batch_size): {batch_size}")
    logger.info(f"  工作线程数 (num_workers): {num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练集打乱顺序
        num_workers=num_workers,
        pin_memory=True,        # 使用锁页内存，加速 GPU 数据传输
        drop_last=False         # 不丢弃最后一个不完整的 batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # 验证集不打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # ========== 12. 构建损失函数 ==========
    logger.info("正在构建损失函数...")
    # 使用 DiceFocalLoss，参数从配置读取
    dice_weight = getattr(cfg.training, 'dice_weight', 0.5)
    focal_weight = getattr(cfg.training, 'focal_weight', 0.5)

    # 从 loss 配置段读取参数
    focal_alpha = getattr(cfg.loss, 'focal_alpha', 0.25)
    focal_gamma = getattr(cfg.loss, 'focal_gamma', 2.0)
    dice_smooth = getattr(cfg.loss, 'dice_smooth', 1.0)

    criterion = DiceFocalLoss(
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        dice_kwargs={'smooth': dice_smooth},
        focal_kwargs={'alpha': focal_alpha, 'gamma': focal_gamma}
    )
    logger.info(f"  损失函数: DiceFocalLoss (dice_weight={dice_weight}, focal_weight={focal_weight})")

    # ========== 13. 构建优化器和学习率调度器 ==========
    logger.info("正在构建优化器...")
    optimizer = build_optimizer(model, cfg)
    logger.info(f"  优化器: {type(optimizer).__name__}, 初始学习率: {cfg.training.lr}")

    logger.info("正在构建学习率调度器...")
    scheduler = build_scheduler(optimizer, cfg, num_epochs=cfg.training.epochs)
    if scheduler is not None:
        logger.info(f"  调度器: {type(scheduler).__name__}")
    else:
        logger.info("  未使用学习率调度器")

    # ========== 14. 构建早停机制 ==========
    patience = getattr(cfg.training, 'patience', 10)
    logger.info(f"早停配置: patience={patience}")
    early_stopping = EarlyStopping(
        patience=patience,
        mode='max',        # 监控 Dice 系数，越高越好
        verbose=True
    )

    # ========== 15. 创建 Trainer 并开始训练 ==========
    logger.info("正在初始化 Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        cfg=cfg,
        logger=logger
    )

    # 开始训练！
    history, trained_model = trainer.fit()

    # ========== 16. 训练完成 ==========
    logger.info("=" * 80)
    logger.info("🎉 训练完成！最佳模型已保存到 checkpoints/best_model.pth")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
