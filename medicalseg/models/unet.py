"""
医学图像分割模型 - U-Net经典网络结构
U-Net是一种编码器-解码器结构的卷积神经网络，通过跳跃连接融合深层语义信息和浅层细节信息，
广泛应用于医学图像分割任务。
论文参考：U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
"""
import torch
import torch.nn as nn

from .layers import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    经典U-Net网络结构
    
    网络结构：
    - 编码器（收缩路径）：4次下采样，每次空间尺寸减半、通道数翻倍
    - 瓶颈层：最底层特征提取
    - 解码器（扩展路径）：4次上采样，每次空间尺寸翻倍、通道数减半，与编码器跳跃连接拼接
    - 输出层：1x1卷积映射到类别数
    
    通道数变化（base_channels=64）：
    - bilinear=False（转置卷积）: in_ch -> 64 -> 128 -> 256 -> 512 -> 1024 -> 512 -> 256 -> 128 -> 64 -> num_classes
    - bilinear=True（双线性插值）:  in_ch -> 64 -> 128 -> 256 -> 512 -> 512  -> 256 -> 128 -> 64  -> 64 -> num_classes
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 1, bilinear: bool = False):
        """
        初始化UNet模型
        
        Args:
            in_channels: 输入图像通道数，灰度图为1，RGB图为3
            num_classes: 分割类别数，二分类为1，多分类按实际数量设置
            bilinear: 是否使用双线性插值上采样，False使用转置卷积
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # 基础通道数
        base_ch = 64
        
        # -------------------- 编码器（收缩路径） --------------------
        # 输入层：不进行下采样，直接对输入图像进行双卷积
        # 输出尺寸与输入相同，通道数：in_channels -> 64
        self.inc = DoubleConv(in_channels, base_ch)
        
        # 下采样层1：空间尺寸减半，通道数翻倍
        # 64 -> 128
        self.down1 = Down(base_ch, base_ch * 2)
        
        # 下采样层2：空间尺寸再减半，通道数再翻倍
        # 128 -> 256
        self.down2 = Down(base_ch * 2, base_ch * 4)
        
        # 下采样层3：空间尺寸再减半，通道数再翻倍
        # 256 -> 512
        self.down3 = Down(base_ch * 4, base_ch * 8)
        
        # 下采样层4（瓶颈层）：最底层
        # 通道数根据bilinear参数决定：
        # - bilinear=False（转置卷积）: 通道数翻倍到1024，转置卷积会将通道数减半
        # - bilinear=True（双线性插值）: 通道数保持512（不翻倍），因为双线性上采样不会改变通道数
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor)
        
        # -------------------- 解码器（扩展路径） --------------------
        # 上采样层1：与down3跳跃连接
        # bilinear=False: x5是1024通道，转置卷积到512，与x4(512)拼接为1024，输出512
        # bilinear=True:  x5是512通道，双线性上采样不变，与x4(512)拼接为1024，输出256
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)
        
        # 上采样层2：与down2跳跃连接
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)
        
        # 上采样层3：与down1跳跃连接
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)
        
        # 上采样层4：与inc跳跃连接
        self.up4 = Up(base_ch * 2, base_ch, bilinear)
        
        # -------------------- 输出层 --------------------
        # 1x1卷积：64 -> num_classes
        self.outc = OutConv(base_ch, num_classes)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"UNet模型参数量: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状 (B, C, H, W)
               B: 批大小
               C: 通道数（in_channels）
               H: 图像高度
               W: 图像宽度
        
        Returns:
            logits: 分割预测logits，形状 (B, num_classes, H, W)
                   注意：输出不经过sigmoid/softmax激活，由损失函数统一处理
        """
        # -------------------- 编码器前向传播 --------------------
        # x1: 输入层特征，保留用于最后一级跳跃连接
        # 形状变化：(B, C, H, W) -> (B, 64, H, W)
        x1 = self.inc(x)
        
        # x2: 第一次下采样特征
        # 形状变化：(B, 64, H, W) -> (B, 128, H/2, W/2)
        x2 = self.down1(x1)
        
        # x3: 第二次下采样特征
        # 形状变化：(B, 128, H/2, W/2) -> (B, 256, H/4, W/4)
        x3 = self.down2(x2)
        
        # x4: 第三次下采样特征
        # 形状变化：(B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
        x4 = self.down3(x3)
        
        # x5: 第四次下采样（瓶颈层）特征
        # bilinear=False: (B, 512, H/8, W/8) -> (B, 1024, H/16, W/16)
        # bilinear=True:  (B, 512, H/8, W/8) -> (B, 512, H/16, W/16)
        x5 = self.down4(x4)
        
        # -------------------- 解码器前向传播 --------------------
        # 第一级上采样：x5上采样后与x4跳跃连接拼接
        # bilinear=False: (B, 1024, H/16, W/16) -> 转置卷积到512，cat x4(512) -> (B, 512, H/8, W/8)
        # bilinear=True:  (B, 512, H/16, W/16) -> 双线性上采样到512，cat x4(512) -> (B, 256, H/8, W/8)
        x = self.up1(x5, x4)
        
        # 第二级上采样：与x3跳跃连接拼接
        # -> (B, 256/128, H/4, W/4)
        x = self.up2(x, x3)
        
        # 第三级上采样：与x2跳跃连接拼接
        # -> (B, 128/64, H/2, W/2)
        x = self.up3(x, x2)
        
        # 第四级上采样：与x1跳跃连接拼接
        # -> (B, 64, H, W)
        x = self.up4(x, x1)
        
        # -------------------- 输出层 --------------------
        # 1x1卷积映射到类别数
        # (B, 64, H, W) -> (B, num_classes, H, W)
        logits = self.outc(x)
        
        return logits
