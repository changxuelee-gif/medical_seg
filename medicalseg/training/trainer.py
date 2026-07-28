"""
医学图像分割训练器核心模块
实现 Trainer 类，负责完整的训练和验证流程，包括：
- 单轮训练（train_one_epoch）
- 验证评估（validate）
- 检查点保存（save_checkpoint）
- 完整训练循环（fit）
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import get_metrics


class Trainer:
    """
    医学图像分割训练器类
    
    封装完整的训练和验证流程，支持：
    - 训练过程中的损失和指标记录
    - 早停机制
    - 学习率调度
    - 最佳模型和最后模型保存
    - 训练历史记录（metrics.json）
    - 配置文件备份
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        early_stopping: Optional[object],
        device: torch.device,
        cfg,
        logger=None
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的分割模型（nn.Module 子类）
            train_loader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
            criterion: 损失函数（nn.Module 子类，如 DiceFocalLoss）
            optimizer: 优化器（如 AdamW）
            scheduler: 学习率调度器（如 CosineAnnealingLR、ReduceLROnPlateau），可为 None
            early_stopping: 早停机制对象，可为 None
            device: 计算设备（torch.device，如 cuda:0 或 cpu）
            cfg: Config 配置对象，包含训练、数据、路径等配置
            logger: 日志记录器对象，若为 None 则使用 print 输出
        """
        # 保存所有组件为实例属性
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = device
        self.cfg = cfg
        self.logger = logger

        # 初始化训练历史记录字典，用于记录每个 epoch 的指标
        self.history = {
            'train_loss': [],      # 训练集损失
            'val_loss': [],        # 验证集损失
            'val_dice': [],        # 验证集 Dice 系数
            'val_iou': [],         # 验证集 IoU
            'val_precision': [],   # 验证集精确率
            'val_recall': [],      # 验证集召回率
            'val_accuracy': [],    # 验证集准确率
            'lr': []               # 当前学习率
        }

        # 获取项目根目录（当前文件向上三级：medicalseg/training/trainer.py -> 项目根）
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent

        # 创建必要的目录（checkpoint_dir、log_dir、output_dir 等）
        # 从配置中获取路径，若未配置则使用默认值
        checkpoint_dir = getattr(cfg.paths, 'checkpoint_dir', 'checkpoints')
        log_dir = getattr(cfg.paths, 'log_dir', 'logs')
        output_dir = getattr(cfg.paths, 'output_dir', 'outputs')

        # 转换为 Path 对象并创建目录（parents=True 递归创建父目录，exist_ok=True 目录已存在不报错）
        self.checkpoint_dir = self.project_root / checkpoint_dir
        self.log_dir = self.project_root / log_dir
        self.output_dir = self.project_root / output_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        """
        内部辅助方法：输出日志信息
        
        如果 logger 存在则使用 logger.info，否则使用 print 直接打印
        
        Args:
            message: 要输出的日志消息
        """
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def train_one_epoch(self, epoch: int) -> float:
        """
        训练单个 epoch
        
        执行流程：
        1. 将模型设置为训练模式（model.train()）
        2. 遍历训练集 DataLoader，逐个 batch 进行前向传播、损失计算、反向传播、参数更新
        3. 使用 tqdm 显示进度条，实时更新当前 loss
        4. 返回该 epoch 的平均训练损失
        
        Args:
            epoch: 当前 epoch 编号（从 0 开始）
            
        Returns:
            avg_train_loss: 该 epoch 的平均训练损失（Python float）
        """
        # 将模型设置为训练模式（启用 BatchNorm、Dropout 等）
        self.model.train()

        # 初始化累计变量
        running_loss = 0.0  # 累计损失
        num_batches = len(self.train_loader)  # batch 总数

        # 使用 tqdm 包裹 train_loader，显示进度条
        # desc 设置进度条前缀，显示当前 epoch/总 epoch
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.cfg.training.epochs}')

        # 遍历每个 batch
        for batch_idx, batch in enumerate(pbar):
            # 解包 batch：images 和 masks
            images, masks = batch

            # 将数据移到指定设备（GPU/CPU）
            # 确保张量形状为 (B, 1, H, W)
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            # 梯度清零（防止梯度累积）
            self.optimizer.zero_grad()

            # 前向传播：模型预测
            outputs = self.model(images)

            # 计算损失
            loss = self.criterion(outputs, masks)

            # 反向传播：计算梯度
            loss.backward()

            # 参数更新：优化器步进
            self.optimizer.step()

            # 累计损失（loss.item() 获取 Python float 值，避免保留计算图）
            running_loss += loss.item()

            # 更新 tqdm 进度条的后置信息，显示当前 batch 的 loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
            })

        # 计算平均训练损失
        avg_train_loss = running_loss / num_batches

        return avg_train_loss

    @torch.no_grad()  # 禁用梯度计算，节省内存和计算资源
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        在验证集上评估模型性能
        
        执行流程：
        1. 将模型设置为评估模式（model.eval()）
        2. 遍历验证集 DataLoader，进行前向传播和损失计算
        3. 使用 get_metrics 计算所有评估指标（Dice、IoU、Precision、Recall、Accuracy）
        4. 累计并计算所有指标的平均值
        
        Args:
            epoch: 当前 epoch 编号（从 0 开始）
            
        Returns:
            (avg_val_loss, metrics_dict):
                - avg_val_loss: 验证集平均损失（Python float）
                - metrics_dict: 包含各指标平均值的字典
                  {'dice': float, 'iou': float, 'precision': float, 'recall': float, 'accuracy': float}
        """
        # 将模型设置为评估模式（禁用 BatchNorm、Dropout 等）
        self.model.eval()

        # 初始化累计变量
        running_loss = 0.0
        # 各指标累计值
        dice_sum = 0.0
        iou_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        accuracy_sum = 0.0
        num_batches = len(self.val_loader)

        # 使用 tqdm 显示验证进度条
        pbar = tqdm(self.val_loader, desc=f'Validation Epoch {epoch+1}', leave=False)

        # 遍历验证集每个 batch
        for batch in pbar:
            images, masks = batch

            # 将数据移到指定设备
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            # 前向传播（torch.no_grad() 已装饰，不会计算梯度）
            outputs = self.model(images)

            # 计算损失
            loss = self.criterion(outputs, masks)

            # 计算所有评估指标
            # get_metrics 返回字典，包含 dice/iou/precision/recall/accuracy
            batch_metrics = get_metrics(outputs, masks)

            # 累计损失和各指标
            running_loss += loss.item()
            dice_sum += batch_metrics['dice']
            iou_sum += batch_metrics['iou']
            precision_sum += batch_metrics['precision']
            recall_sum += batch_metrics['recall']
            accuracy_sum += batch_metrics['accuracy']

            # 更新进度条信息
            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'dice': f'{batch_metrics["dice"]:.4f}'
            })

        # 计算平均值
        avg_val_loss = running_loss / num_batches
        metrics_dict = {
            'dice': dice_sum / num_batches,
            'iou': iou_sum / num_batches,
            'precision': precision_sum / num_batches,
            'recall': recall_sum / num_batches,
            'accuracy': accuracy_sum / num_batches
        }

        return avg_val_loss, metrics_dict

    def save_checkpoint(self, path: Path) -> None:
        """
        保存模型检查点
        
        保存内容包括：
        - 模型权重（model_state_dict）
        - 优化器状态（optimizer_state_dict）
        - 完整配置（config）
        - 训练历史（history）
        
        Args:
            path: 检查点保存路径（Path 对象）
        """
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        # 构建要保存的检查点字典
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'history': self.history,
        }

        # 如果有学习率调度器，也保存其状态
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存检查点
        torch.save(checkpoint, path)
        self._log(f"检查点已保存至: {path}")

    def fit(self) -> Tuple[Dict[str, list], nn.Module]:
        """
        完整训练循环（主入口方法）
        
        执行流程：
        1. 从 epoch 0 到 epochs-1 循环训练
        2. 每个 epoch：
           a. 调用 train_one_epoch 训练一轮
           b. 调用 validate 在验证集上评估
           c. 获取当前学习率
           d. 将所有指标记录到 history
           e. 格式化打印当前 epoch 的所有指标（对齐美观）
           f. 早停检查：传入 val_dice 和 model，若改进则保存最佳模型
           g. 学习率调度器步进（plateau 类型需要传入指标值）
           h. 若触发早停，打印信息并 break
        3. 训练结束后恢复最佳模型权重
        4. 保存最后模型到 checkpoints/last_model.pth
        5. 保存训练历史到 logs/metrics.json（处理 tensor 序列化问题）
        6. 复制当前配置到 checkpoints/config.yaml
        7. 返回 history 和训练好的 model
        
        Returns:
            (self.history, self.model):
                - self.history: 训练历史记录字典
                - self.model: 训练完成的模型（已加载最佳权重）
        """
        # 打印训练开始信息
        self._log("=" * 80)
        self._log("开始训练")
        self._log("=" * 80)
        self._log(f"总训练轮数: {self.cfg.training.epochs}")
        self._log(f"训练集批次数: {len(self.train_loader)}")
        self._log(f"验证集批次数: {len(self.val_loader)}")
        self._log(f"设备: {self.device}")
        self._log("=" * 80)

        # 定义最佳模型保存路径
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        last_model_path = self.checkpoint_dir / 'last_model.pth'
        metrics_path = self.log_dir / 'metrics.json'
        config_save_path = self.checkpoint_dir / 'config.yaml'

        # 主训练循环
        for epoch in range(self.cfg.training.epochs):
            # ---------- 1. 训练一个 epoch ----------
            train_loss = self.train_one_epoch(epoch)

            # ---------- 2. 在验证集上评估 ----------
            val_loss, val_metrics = self.validate(epoch)

            # ---------- 3. 获取当前学习率 ----------
            current_lr = self.optimizer.param_groups[0]['lr']

            # ---------- 4. 记录到 history ----------
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)

            # ---------- 5. 格式化打印当前 epoch 的所有指标 ----------
            # 打印分隔线
            self._log("-" * 80)
            self._log(f"Epoch [{epoch+1}/{self.cfg.training.epochs}] 训练结果:")
            self._log(f"  训练损失 (Train Loss):     {train_loss:.4f}")
            self._log(f"  验证损失 (Val Loss):       {val_loss:.4f}")
            self._log(f"  Dice 系数:                 {val_metrics['dice']:.4f}")
            self._log(f"  IoU 交并比:                {val_metrics['iou']:.4f}")
            self._log(f"  精确率 (Precision):        {val_metrics['precision']:.4f}")
            self._log(f"  召回率 (Recall):           {val_metrics['recall']:.4f}")
            self._log(f"  准确率 (Accuracy):         {val_metrics['accuracy']:.4f}")
            self._log(f"  当前学习率 (LR):           {current_lr:.6f}")
            self._log("-" * 80)

            # ---------- 6. EarlyStopping 早停检查 ----------
            if self.early_stopping is not None:
                # 传入 val_dice（因为 Dice 是越高越好的指标，mode='max'）
                # EarlyStopping 内部会判断是否改进，若改进则深拷贝模型权重到内存
                # 注意：这里我们需要自己保存最佳模型到磁盘
                prev_best_score = self.early_stopping.best_score
                self.early_stopping(val_metrics['dice'], self.model)

                # 如果当前模型是最佳模型（best_score 更新了），保存到磁盘
                if prev_best_score is None or val_metrics['dice'] > prev_best_score:
                    self.save_checkpoint(best_model_path)

            # ---------- 7. 学习率调度器步进 ----------
            if self.scheduler is not None:
                # 判断是否为 ReduceLROnPlateau 类型（需要传入指标值）
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Plateau 调度器根据验证指标调整学习率，传入 val_dice（越高越好）
                    self.scheduler.step(val_metrics['dice'])
                else:
                    # 其他调度器（CosineAnnealingLR、StepLR 等）直接 step()
                    self.scheduler.step()

            # ---------- 8. 检查是否触发早停 ----------
            if self.early_stopping is not None and self.early_stopping.early_stop:
                self._log("=" * 80)
                self._log(f"早停触发！已连续 {self.early_stopping.patience} 轮验证指标无改进。")
                self._log(f"最佳验证 Dice 系数: {self.early_stopping.best_score:.4f}")
                self._log("=" * 80)
                break

        # ---------- 训练结束 ----------
        self._log("=" * 80)
        self._log("训练循环结束，正在进行收尾工作...")

        # 9. 恢复最佳模型权重
        if self.early_stopping is not None:
            self.model = self.early_stopping.load_best_model(self.model)

        # 10. 保存最后一轮的模型
        self.save_checkpoint(last_model_path)

        # 11. 保存训练历史到 logs/metrics.json
        # 需要将所有数值转换为 Python float（处理可能的 tensor/numpy 类型）
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = []
            for v in values:
                # 如果是 torch tensor，先转为 numpy 再转 float
                if isinstance(v, torch.Tensor):
                    serializable_history[key].append(float(v.cpu().numpy()))
                # 如果是 numpy 数值，转 float
                elif hasattr(v, 'item'):
                    serializable_history[key].append(v.item())
                # 否则直接转 float（处理 Python 原生数值）
                else:
                    serializable_history[key].append(float(v))

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        self._log(f"训练指标历史已保存至: {metrics_path}")

        # 12. 复制当前配置到 checkpoints/config.yaml
        import yaml
        # 将 Config 对象转为普通字典
        config_dict = self.cfg.to_dict() if hasattr(self.cfg, 'to_dict') else dict(self.cfg)
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._log(f"配置文件已备份至: {config_save_path}")

        self._log("=" * 80)
        self._log("训练完成！")
        self._log(f"最佳模型已保存至: {best_model_path}")
        self._log(f"最后模型已保存至: {last_model_path}")
        self._log("=" * 80)

        return self.history, self.model
