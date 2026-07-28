"""
医学图像分割模型 - 基础层模块
包含U-Net和UNet++共用的基础组件：DoubleConv、Down、Up、OutConv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    两次卷积模块：(Conv2d -> BatchNorm2d -> ReLU) * 2
    
    这是U-Net中最基础的卷积块，连续应用两次3x3卷积，每次卷积后接批归一化和ReLU激活。
    使用same卷积（kernel_size=3, padding=1）保持特征图尺寸不变。
    """
    
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        """
        初始化DoubleConv模块
        
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            mid_ch: 中间通道数，如果为None则等于out_ch
        """
        super().__init__()
        
        # 如果未指定中间通道数，则中间通道数等于输出通道数
        if mid_ch is None:
            mid_ch = out_ch
            
        # 构建连续两次卷积的序列
        self.double_conv = nn.Sequential(
            # 第一次卷积：in_ch -> mid_ch
            # kernel_size=3, padding=1 实现same卷积，保持空间尺寸不变
            # bias=False：因为后面接BatchNorm，bias会被BN抵消，无需额外参数
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            
            # 第二次卷积：mid_ch -> out_ch
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块：MaxPool2d(2) -> DoubleConv
    
    编码器路径中的下采样块，先通过2x2最大池化将空间尺寸减半，
    然后通过DoubleConv提取特征并将通道数翻倍。
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        """
        初始化Down模块
        
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
        """
        super().__init__()
        
        # 构建下采样序列：先最大池化，再双卷积
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2最大池化，空间尺寸减半
            DoubleConv(in_ch, out_ch)                # 双卷积提取特征
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块：上采样 -> 与跳跃连接特征拼接 -> DoubleConv
    
    解码器路径中的上采样块，将深层特征上采样到与浅层跳跃连接特征相同的尺寸，
    然后在通道维度拼接，最后通过DoubleConv融合特征。
    支持两种上采样方式：
    1. 双线性插值（bilinear=True）：使用nn.Upsample，参数少，上采样更平滑
    2. 转置卷积（bilinear=False）：使用nn.ConvTranspose2d，可学习上采样参数
    """
    
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        """
        初始化Up模块
        
        Args:
            in_ch: 输入通道数（注意：拼接后通道数会变为in_ch）
            out_ch: 输出通道数
            bilinear: 是否使用双线性插值上采样，False则使用转置卷积
        """
        super().__init__()
        
        # 根据bilinear参数选择上采样方式
        if bilinear:
            # 双线性插值上采样：
            # scale_factor=2：将空间尺寸放大2倍
            # mode='bilinear'：双线性插值
            # align_corners=True：对齐角点像素，使上采样结果更准确
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 双线性上采样不改变通道数：
            # 输入x1的通道数是in_ch//2（瓶颈层通道数在bilinear时减半）
            # 上采样后还是in_ch//2，与跳跃连接x2（通道数in_ch//2）拼接后是in_ch
            # mid_ch设为in_ch//2，减少参数量
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            # 转置卷积上采样（反卷积）：
            # in_ch -> in_ch//2：输入通道数in_ch（来自上一层），输出通道数in_ch//2
            # kernel_size=2, stride=2：实现2倍上采样，空间尺寸翻倍
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            # 转置卷积后通道数为in_ch//2，与跳跃连接（通道数in_ch//2）拼接后是in_ch
            self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x1: 深层特征图，需要上采样
            x2: 浅层跳跃连接特征图，来自编码器对应层
        
        Returns:
            融合后的特征图
        """
        # 第一步：对深层特征进行上采样
        x1 = self.up(x1)
        
        # 第二步：处理尺寸不匹配问题
        # 由于输入图像尺寸可能不是2的整数次幂，下采样再上采样后可能出现尺寸差异
        # 计算两个特征图的尺寸差异
        # x2是跳跃连接特征，尺寸更大；x1是上采样后的特征
        diffY = x2.size()[2] - x1.size()[2]  # 高度差
        diffX = x2.size()[3] - x1.size()[3]  # 宽度差
        
        # 使用F.pad对x1进行填充，使其尺寸与x2一致
        # pad参数格式：[左, 右, 上, 下]
        # diffX // 2：左侧填充
        # diffX - diffX // 2：右侧填充（处理奇数差异）
        # diffY // 2：上方填充
        # diffY - diffY // 2：下方填充
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        
        # 第三步：在通道维度拼接特征
        # x1经过上采样，通道数为in_ch//2；x2来自编码器，通道数也是in_ch//2
        # 拼接后通道数为in_ch
        x = torch.cat([x2, x1], dim=1)
        
        # 第四步：通过双卷积融合拼接后的特征
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层：1x1卷积
    
    使用1x1卷积将最终特征图的通道数调整为分割类别数，输出每个像素的类别预测logits。
    1x1卷积不改变空间尺寸，只在通道维度进行线性组合。
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        """
        初始化OutConv模块
        
        Args:
            in_ch: 输入通道数（与上一层输出通道一致）
            out_ch: 输出通道数（等于分割类别数num_classes）
        """
        super().__init__()
        # 1x1卷积，kernel_size=1，无padding
        # 用于将通道数映射到类别数
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回logits（不经过sigmoid/softmax，由损失函数处理）"""
        return self.conv(x)
