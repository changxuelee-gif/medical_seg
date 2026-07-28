"""
医学图像分割评估指标模块
包含 Dice 系数、IoU、精确率、召回率、准确率等常用分割评估指标。
所有指标均为普通函数（非 nn.Module），接收 logits 和 targets，返回 Python float 标量。
兼容 UNet++ 深度监督模式：若 logits 为 list，则取最后一个输出计算指标。
所有计算使用 PyTorch 张量操作，保留 GPU 计算能力，不转换为 numpy。
"""
import torch
from typing import Dict


def _prepare_inputs(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    内部辅助函数：预处理输入张量
    
    处理流程：
    1. 若 logits 为 list/tuple（深度监督模式），取最后一个输出作为最终预测
    2. 对 logits 做 sigmoid 激活得到概率
    3. 按阈值二值化得到预测掩码（0 或 1）
    4. 将预测和标签展平，便于计算混淆矩阵元素
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W) 或深度监督的 list
        targets: 真实掩码，形状 (B, 1, H, W)，值为 0 或 1
        threshold: 二值化阈值，sigmoid 后概率 > threshold 判定为正类（前景）
    
    Returns:
        pred_flat: 二值化后的预测，形状 (B, H*W)，值为 0 或 1
        targets_flat: 展平后的真实标签，形状 (B, H*W)，值为 0 或 1
    """
    # 处理深度监督模式：取最后一个输出（最深层监督分支，通常是 x^{0,4}）
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    
    # sigmoid 激活得到概率
    probs = torch.sigmoid(logits)
    
    # 二值化：概率大于阈值为 1（前景），否则为 0（背景）
    preds = (probs > threshold).float()
    
    # 展平空间维度：(B, 1, H, W) -> (B, H*W)
    pred_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    return pred_flat, targets_flat


def _compute_confusion_matrix(pred_flat: torch.Tensor, targets_flat: torch.Tensor):
    """
    内部辅助函数：计算混淆矩阵元素 TP、FP、TN、FN
    
    Args:
        pred_flat: 二值化预测，形状 (B, N)，N = H*W
        targets_flat: 真实标签，形状 (B, N)
    
    Returns:
        tp: 真阳性（True Positive），正确预测为前景的像素数，形状 (B,)
        fp: 假阳性（False Positive），错误预测为前景的像素数（背景被判为前景），形状 (B,)
        tn: 真阴性（True Negative），正确预测为背景的像素数，形状 (B,)
        fn: 假阴性（False Negative），错误预测为背景的像素数（前景被判为背景），形状 (B,)
    """
    # TP：预测为 1 且 真实为 1
    tp = (pred_flat * targets_flat).sum(dim=1)
    
    # FP：预测为 1 但 真实为 0
    fp = (pred_flat * (1 - targets_flat)).sum(dim=1)
    
    # TN：预测为 0 且 真实为 0
    tn = ((1 - pred_flat) * (1 - targets_flat)).sum(dim=1)
    
    # FN：预测为 0 但 真实为 1
    fn = ((1 - pred_flat) * targets_flat).sum(dim=1)
    
    return tp, fp, tn, fn


def dice_coeff(logits: torch.Tensor, 
               targets: torch.Tensor, 
               smooth: float = 1.0, 
               threshold: float = 0.5) -> float:
    """
    计算 Dice 系数（F1-Score）
    
    Dice 系数衡量预测区域与真实区域的重叠程度，是分割任务最常用的评估指标之一。
    取值范围 [0, 1]，1 表示完全重叠，0 表示完全不重叠。
    
    公式：
        Dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W)，未 sigmoid
        targets: 真实掩码，形状 (B, 1, H, W)，值为 0/1
        smooth: 平滑因子，防止除零，默认 1.0
        threshold: 二值化阈值，默认 0.5
    
    Returns:
        Dice 系数，Python float 标量，按 batch 平均
    """
    # 预处理：二值化 + 展平
    pred_flat, targets_flat = _prepare_inputs(logits, targets, threshold)
    
    # 计算混淆矩阵元素
    tp, fp, tn, fn = _compute_confusion_matrix(pred_flat, targets_flat)
    
    # 按样本计算 Dice 系数
    # 分母 = 2*TP + FP + FN = TP + (TP + FP + FN) = 预测正例数 + 真实正例数（交集并集相关）
    dice_per_sample = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    
    # 按 batch 平均，转为 Python float
    return dice_per_sample.mean().item()


def iou_score(logits: torch.Tensor, 
              targets: torch.Tensor, 
              smooth: float = 1.0, 
              threshold: float = 0.5) -> float:
    """
    计算 IoU（Intersection over Union，交并比 / Jaccard Index）
    
    IoU 衡量预测区域与真实区域的交集占并集的比例。
    取值范围 [0, 1]，1 表示完全重叠，0 表示完全不重叠。
    与 Dice 的关系：IoU = Dice / (2 - Dice)
    
    公式：
        IoU = (TP + smooth) / (TP + FP + FN + smooth)
        即：交集 / 并集
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W)
        targets: 真实掩码，形状 (B, 1, H, W)
        smooth: 平滑因子，防止除零，默认 1.0
        threshold: 二值化阈值，默认 0.5
    
    Returns:
        IoU 分数，Python float 标量，按 batch 平均
    """
    pred_flat, targets_flat = _prepare_inputs(logits, targets, threshold)
    tp, fp, tn, fn = _compute_confusion_matrix(pred_flat, targets_flat)
    
    # 交集 = TP，并集 = TP + FP + FN
    iou_per_sample = (tp + smooth) / (tp + fp + fn + smooth)
    
    return iou_per_sample.mean().item()


def precision(logits: torch.Tensor, 
              targets: torch.Tensor, 
              threshold: float = 0.5, 
              smooth: float = 1e-6) -> float:
    """
    计算精确率（Precision / 查准率）
    
    精确率表示在所有被预测为前景的像素中，真正是前景的比例。
    取值范围 [0, 1]，越高表示误检（假阳性）越少。
    
    公式：
        Precision = TP / (TP + FP + smooth)
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W)
        targets: 真实掩码，形状 (B, 1, H, W)
        threshold: 二值化阈值，默认 0.5
        smooth: 平滑因子，防止除零，默认 1e-6（很小的值，不显著影响结果）
    
    Returns:
        精确率，Python float 标量，按 batch 平均
    """
    pred_flat, targets_flat = _prepare_inputs(logits, targets, threshold)
    tp, fp, tn, fn = _compute_confusion_matrix(pred_flat, targets_flat)
    
    # 精确率 = TP / (TP + FP)
    precision_per_sample = tp / (tp + fp + smooth)
    
    return precision_per_sample.mean().item()


def recall(logits: torch.Tensor, 
           targets: torch.Tensor, 
           threshold: float = 0.5, 
           smooth: float = 1e-6) -> float:
    """
    计算召回率（Recall / 灵敏度 / Sensitivity / 查全率）
    
    召回率表示在所有真实前景像素中，被正确预测为前景的比例。
    取值范围 [0, 1]，越高表示漏检（假阴性）越少。
    
    公式：
        Recall = TP / (TP + FN + smooth)
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W)
        targets: 真实掩码，形状 (B, 1, H, W)
        threshold: 二值化阈值，默认 0.5
        smooth: 平滑因子，防止除零，默认 1e-6
    
    Returns:
        召回率，Python float 标量，按 batch 平均
    """
    pred_flat, targets_flat = _prepare_inputs(logits, targets, threshold)
    tp, fp, tn, fn = _compute_confusion_matrix(pred_flat, targets_flat)
    
    # 召回率 = TP / (TP + FN)
    recall_per_sample = tp / (tp + fn + smooth)
    
    return recall_per_sample.mean().item()


def accuracy(logits: torch.Tensor, 
             targets: torch.Tensor, 
             threshold: float = 0.5) -> float:
    """
    计算准确率（Accuracy）
    
    准确率表示所有像素中被正确分类的比例（包括前景和背景）。
    取值范围 [0, 1]，越高表示整体分类越准确。
    注意：在类别极度不平衡时（如前景很小），准确率可能虚高，需结合 Dice/IoU 综合判断。
    
    公式：
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        即：正确预测数 / 总像素数
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W)
        targets: 真实掩码，形状 (B, 1, H, W)
        threshold: 二值化阈值，默认 0.5
    
    Returns:
        准确率，Python float 标量，按 batch 平均
    """
    pred_flat, targets_flat = _prepare_inputs(logits, targets, threshold)
    tp, fp, tn, fn = _compute_confusion_matrix(pred_flat, targets_flat)
    
    # 准确率 = (TP + TN) / (TP + TN + FP + FN)
    # 分母即总像素数 = tp + fp + tn + fn
    acc_per_sample = (tp + tn) / (tp + tn + fp + fn)
    
    return acc_per_sample.mean().item()


def get_metrics(logits: torch.Tensor, 
                targets: torch.Tensor, 
                threshold: float = 0.5) -> Dict[str, float]:
    """
    一次性计算所有评估指标，返回字典
    
    方便训练/验证循环中调用，一次前向传播即可获得全部指标。
    
    Args:
        logits: 模型输出，形状 (B, 1, H, W)，未 sigmoid
                若为深度监督 list，内部取最后一个输出
        targets: 真实掩码，形状 (B, 1, H, W)，值为 0/1
        threshold: 二值化阈值，默认 0.5
    
    Returns:
        包含所有指标的字典：
        {
            'dice': float,      # Dice 系数
            'iou': float,       # IoU 交并比
            'precision': float, # 精确率
            'recall': float,    # 召回率
            'accuracy': float   # 准确率
        }
    """
    return {
        'dice': dice_coeff(logits, targets, threshold=threshold),
        'iou': iou_score(logits, targets, threshold=threshold),
        'precision': precision(logits, targets, threshold=threshold),
        'recall': recall(logits, targets, threshold=threshold),
        'accuracy': accuracy(logits, targets, threshold=threshold),
    }
