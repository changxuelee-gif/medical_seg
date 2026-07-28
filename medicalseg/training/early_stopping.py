"""
医学图像分割训练 - 早停机制模块
本模块提供 EarlyStopping 类，用于在训练过程中监控验证集指标，
当指标在指定轮数内没有提升时提前停止训练，防止过拟合。
"""
import copy
import torch
import torch.nn as nn
from typing import Optional


class EarlyStopping:
    """
    早停机制类：监控验证集指标，在指标长期不提升时提前终止训练。

    早停（Early Stopping）是一种正则化技术，通过监控验证集上的性能指标，
    当指标连续多轮没有显著改善时停止训练，从而：
    1. 防止模型过拟合
    2. 节省训练时间和计算资源
    3. 自动保存验证集上表现最好的模型权重

    使用示例：
        early_stopping = EarlyStopping(patience=10, mode='max', verbose=True)
        for epoch in range(num_epochs):
            train(...)
            val_metric = validate(...)
            early_stopping(val_metric, model)
            if early_stopping.early_stop:
                print("早停触发！")
                break
        model = early_stopping.load_best_model(model)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        初始化早停机制。

        参数说明：
            patience (int): 等待多少个 epoch 指标无改进则停止训练。
                例如 patience=10 表示如果连续 10 轮验证指标都没有提升，则触发早停。
                默认值：10
            min_delta (float): 最小改进量，只有当指标改进超过此值时才被认为是真正的改进。
                这可以防止因微小波动导致的频繁计数器重置。
                - 当 mode='max' 时：新 score > best_score + min_delta 才算改进
                - 当 mode='min' 时：新 score < best_score - min_delta 才算改进
                默认值：0.0（任何提升都算改进）
            mode (str): 指标优化方向，可选值：
                - 'max': 指标越高越好（例如 Dice 系数、IoU、准确率等）
                - 'min': 指标越低越好（例如损失值 loss、验证误差等）
                默认值：'max'
            verbose (bool): 是否打印早停过程的详细信息（当前计数器、最佳分数等）。
                默认值：True
        """
        # 验证 mode 参数的合法性
        if mode not in ['max', 'min']:
            raise ValueError(f"mode 参数必须是 'max' 或 'min'，但得到了 '{mode}'")

        # 保存配置参数
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # 初始化内部状态变量
        self.counter = 0  # 连续无改进的轮数计数器
        self.best_score: Optional[float] = None  # 历史最佳分数
        self.early_stop: bool = False  # 是否触发早停标志
        self.best_model_state: Optional[dict] = None  # 最佳模型的 state_dict（深拷贝保存在内存中）

    def __call__(self, score: float, model: nn.Module) -> None:
        """
        每次验证后调用此方法，更新早停状态。

        参数说明：
            score (float): 当前轮次验证集上的监控指标值。
                例如：Dice 系数（mode='max'）或验证损失（mode='min'）。
            model (nn.Module): 当前训练的模型实例。
                当检测到新的最佳分数时，会深拷贝此模型的 state_dict 保存到内存中。

        处理逻辑：
            1. 如果是第一次调用，直接将当前 score 作为 best_score，保存模型权重
            2. 如果不是第一次调用，根据 mode 判断 score 是否有改进：
               - mode='max'：检查 score > best_score + min_delta
               - mode='min'：检查 score < best_score - min_delta
            3. 如果有改进：
               - 更新 best_score 为当前 score
               - 深拷贝模型的 state_dict 保存到 best_model_state
               - 将 counter 重置为 0
            4. 如果没有改进：
               - counter 加 1
               - 如果 counter >= patience，设置 early_stop = True
            5. 如果 verbose=True，打印当前计数器和最佳分数信息
        """
        # 第一次调用，初始化最佳分数
        if self.best_score is None:
            self.best_score = score
            # 深拷贝模型权重，避免后续训练修改了引用的张量
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"[早停] 初始模型已保存，初始分数: {score:.6f}")
            return

        # 判断当前分数是否比最佳分数有改进
        is_improved = False
        if self.mode == 'max':
            # 对于最大化指标（如 Dice），分数越高越好
            # 只有当新分数超过历史最佳 + min_delta 时才算改进
            if score > self.best_score + self.min_delta:
                is_improved = True
        else:  # mode == 'min'
            # 对于最小化指标（如 loss），分数越低越好
            # 只有当新分数低于历史最佳 - min_delta 时才算改进
            if score < self.best_score - self.min_delta:
                is_improved = True

        if is_improved:
            # 有改进：更新最佳分数、保存模型、重置计数器
            if self.verbose:
                print(f"[早停] 验证指标改进: {self.best_score:.6f} --> {score:.6f}")
            self.best_score = score
            # 深拷贝模型权重到内存中
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            # 没有改进：计数器加 1
            self.counter += 1
            if self.verbose:
                print(f"[早停] 验证指标未改进，当前计数器: {self.counter}/{self.patience}，最佳分数: {self.best_score:.6f}")

            # 检查是否达到 patience 阈值，触发早停
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[早停] 已连续 {self.patience} 轮无改进，触发早停！最佳分数: {self.best_score:.6f}")

    def load_best_model(self, model: nn.Module) -> nn.Module:
        """
        将训练过程中保存的最佳模型权重加载回模型。

        参数说明：
            model (nn.Module): 需要加载权重的模型实例。
                通常是训练时使用的同一个模型对象。

        返回：
            nn.Module: 加载了最佳权重后的模型实例（原地修改并返回同一对象）。

        注意：
            此方法应该在训练结束（无论是正常结束还是早停触发）后调用，
            以确保最终使用的是验证集上表现最好的模型权重，而不是最后一轮的权重。
            如果从未保存过最佳模型（best_model_state 为 None），则直接返回原模型。
        """
        if self.best_model_state is not None:
            # 将保存的最佳权重加载到模型中
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"[早停] 已加载最佳模型权重，最佳分数: {self.best_score:.6f}")
        else:
            if self.verbose:
                print("[早停] 警告：没有保存的最佳模型权重，返回原模型")
        return model
