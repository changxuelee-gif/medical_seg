"""
医学图像分割损失函数模块
包含 DiceLoss、FocalLoss 和 DiceFocalLoss 三种常用损失函数。
所有损失函数均兼容 UNet++ 的深度监督（deep_supervision）模式：
- 若输入 logits 为 list，则对每个输出计算 loss 后取平均。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss 损失函数
    
    Dice 系数用于衡量两个集合的相似度，在分割任务中广泛使用。
    DiceLoss = 1 - Dice系数，优化目标是最小化该损失即最大化 Dice 系数。
    
    公式：
        Dice = (2 * intersection + smooth) / (sum(probs^p) + sum(targets^p) + smooth)
        DiceLoss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1.0, p: int = 2):
        """
        初始化 Dice Loss
        
        Args:
            smooth: 平滑因子，防止除零，同时稳定梯度，默认 1.0
            p: 指数参数，p=2 时为软 Dice（使用概率的平方），p=1 时为标准 Dice
        """
        super().__init__()
        self.smooth = smooth
        self.p = p
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Dice Loss
        
        Args:
            logits: 模型输出，形状 (B, 1, H, W)，未经过 sigmoid
                    若为深度监督模式，则为 list，每个元素形状 (B, 1, H, W)
            targets: 真实掩码，形状 (B, 1, H, W)，值为 0 或 1
        
        Returns:
            Dice Loss 标量张量
        """
        # 处理深度监督模式：logits 为 list 时，计算每个输出的 loss 后取平均
        if isinstance(logits, (list, tuple)):
            loss = 0.0
            for logit in logits:
                loss += self._compute_single_loss(logit, targets)
            return loss / len(logits)
        else:
            return self._compute_single_loss(logits, targets)
    
    def _compute_single_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算单个输出的 Dice Loss
        
        Args:
            logits: 模型输出，形状 (B, 1, H, W)
            targets: 真实掩码，形状 (B, 1, H, W)
        
        Returns:
            该输出对应的 Dice Loss
        """
        # 对 logits 做 sigmoid 激活，得到概率值 probs ∈ (0, 1)
        probs = torch.sigmoid(logits)
        
        # 将预测概率和真实标签展平，便于按批次计算
        # 形状从 (B, 1, H, W) 变为 (B, H*W)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # 计算交集：probs 和 targets 逐元素相乘后按空间维度求和
        # intersection 形状：(B,)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        
        # 计算 probs 的 p 次幂和（模长的 p 次方）以及 targets 的 p 次幂和
        # 使用 p=2 相当于对概率做平方，更关注高置信度区域
        probs_sum = (probs_flat ** self.p).sum(dim=1)
        targets_sum = (targets_flat ** self.p).sum(dim=1)
        
        # 计算每个样本的 Dice 系数
        # dice_per_sample 形状：(B,)
        dice_per_sample = (2.0 * intersection + self.smooth) / (probs_sum + targets_sum + self.smooth)
        
        # Dice Loss = 1 - Dice 系数，然后按 batch 取平均
        dice_loss = 1.0 - dice_per_sample.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss 损失函数
    
    Focal Loss 在标准 BCE Loss 基础上进行改进，解决正负样本不平衡和难样本聚焦问题：
    - alpha：平衡正负样本权重，alpha 越大，正样本权重越高
    - gamma：聚焦参数，gamma > 0 时，降低易分样本的权重，使模型更关注难分样本
    
    公式（以 BCEWithLogitsLoss 为基础）：
        pt = 预测正确的概率（正样本时 p，负样本时 1-p）
        Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    
    医疗影像分割中常用参数：alpha=0.8（给正样本更高权重），gamma=2.0
    """
    
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化 Focal Loss
        
        Args:
            alpha: 正样本权重因子，范围 (0, 1)，默认 0.8
                   alpha > 0.5 时正样本权重更高，适合前景像素少的医学图像
            gamma: 聚焦参数，默认 2.0
                   gamma=0 时退化为标准加权 BCE Loss
                   gamma 越大，对难样本的聚焦越强
            reduction: 损失聚合方式，'mean'（默认）或 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # 使用 BCEWithLogitsLoss 作为基础，内部包含 sigmoid 计算，数值更稳定
        # reduction='none' 以便逐元素计算 focal 调制因子
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Focal Loss
        
        Args:
            logits: 模型输出，形状 (B, 1, H, W)，未经过 sigmoid
                    若为深度监督模式，则为 list
            targets: 真实掩码，形状 (B, 1, H, W)，值为 0 或 1
        
        Returns:
            Focal Loss 标量张量
        """
        # 处理深度监督模式
        if isinstance(logits, (list, tuple)):
            loss = 0.0
            for logit in logits:
                loss += self._compute_single_loss(logit, targets)
            return loss / len(logits)
        else:
            return self._compute_single_loss(logits, targets)
    
    def _compute_single_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算单个输出的 Focal Loss
        
        Args:
            logits: 模型输出，形状 (B, 1, H, W)
            targets: 真实掩码，形状 (B, 1, H, W)
        
        Returns:
            该输出对应的 Focal Loss
        """
        # 计算逐元素的 BCE Loss（未做 reduction）
        # bce_loss 形状：(B, 1, H, W)
        bce_loss = self.bce(logits, targets)
        
        # 计算预测概率 probs = sigmoid(logits)
        probs = torch.sigmoid(logits)
        
        # 计算 pt：预测正确的概率
        # 对于正样本（targets=1），pt = probs
        # 对于负样本（targets=0），pt = 1 - probs
        # 使用 torch.where 逐元素选择
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        
        # 计算 alpha 权重：
        # 正样本权重为 alpha，负样本权重为 (1 - alpha)
        alpha_t = torch.where(targets == 1, 
                              torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
                              torch.tensor(1.0 - self.alpha, device=logits.device, dtype=logits.dtype))
        
        # 计算 Focal Loss 的调制因子：(1 - pt)^gamma
        # 对于易分样本（pt 接近 1），调制因子接近 0，损失被降低
        # 对于难分样本（pt 接近 0.5），调制因子接近 1，损失保留
        focal_weight = (1.0 - pt) ** self.gamma
        
        # 组合得到逐元素的 Focal Loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # 按指定方式聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Dice + Focal 组合损失函数
    
    将 Dice Loss 和 Focal Loss 按权重加权组合，兼顾区域重叠度优化和像素级分类优化。
    Dice Loss 关注整体区域重叠，Focal Loss 关注难分像素和类别平衡。
    
    公式：
        Total Loss = dice_weight * DiceLoss + focal_weight * FocalLoss
    
    参考医疗影像分割最佳实践，通常 dice_weight=0.5, focal_weight=0.5。
    两个权重可通过配置文件调整，无需两者之和为 1。
    """
    
    def __init__(self, 
                 dice_weight: float = 0.5, 
                 focal_weight: float = 0.5,
                 dice_kwargs: dict = None,
                 focal_kwargs: dict = None,
                 **kwargs):
        """
        初始化 DiceFocalLoss
        
        Args:
            dice_weight: Dice Loss 的权重，默认 0.5
            focal_weight: Focal Loss 的权重，默认 0.5
            dice_kwargs: 传递给 DiceLoss 的参数字典，如 smooth, p
            focal_kwargs: 传递给 FocalLoss 的参数字典，如 alpha, gamma, reduction
            **kwargs: 兼容配置文件直接传参，可包含 dice 和 focal 的参数，
                     以 dice_ 或 focal_ 前缀区分，例如 dice_smooth=1.0, focal_alpha=0.8
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # 处理 dice_kwargs
        dice_params = dice_kwargs if dice_kwargs is not None else {}
        # 从 kwargs 中提取以 dice_ 开头的参数
        for key, value in kwargs.items():
            if key.startswith('dice_'):
                dice_params[key[5:]] = value
        
        # 处理 focal_kwargs
        focal_params = focal_kwargs if focal_kwargs is not None else {}
        # 从 kwargs 中提取以 focal_ 开头的参数
        for key, value in kwargs.items():
            if key.startswith('focal_'):
                focal_params[key[6:]] = value
        
        # 创建子损失函数实例
        self.dice_loss = DiceLoss(**dice_params)
        self.focal_loss = FocalLoss(**focal_params)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算组合损失
        
        Args:
            logits: 模型输出，形状 (B, 1, H, W)，或深度监督模式的 list
            targets: 真实掩码，形状 (B, 1, H, W)
        
        Returns:
            加权组合后的损失标量
        """
        # 分别计算 Dice Loss 和 Focal Loss
        # 两者内部都已处理深度监督模式
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        
        # 按权重加权求和
        total_loss = self.dice_weight * d_loss + self.focal_weight * f_loss
        
        return total_loss
