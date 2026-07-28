"""
医学图像分割模型 - UNet++ (Nested U-Net) 网络结构
UNet++ 通过嵌套的密集跳跃连接改进了原始 U-Net，减少了编码器和解码器之间的语义鸿沟，
使优化更容易，尤其在医学图像分割任务上表现优异。
论文参考：UNet++: A Nested U-Net Architecture for Medical Image Segmentation (Zhou et al., 2018)

网络结构说明：
- 节点表示为 x^{i,j}：
  - i: 下采样层级索引 (0-4)，i=0为最浅层（原始分辨率），i=4为最深层（瓶颈层）
  - j: 跳跃连接嵌套层级索引 (0-4)，j=0为编码器路径，j>0为嵌套解码节点
- 下采样路径（j=0）：x^{0,0} → x^{1,0} → x^{2,0} → x^{3,0} → x^{4,0}，与U-Net编码器相同
- 嵌套跳跃连接：
  - x^{i,j} 的输入：上采样的 x^{i+1,j-1} + 拼接的 [x^{i,0}, x^{i,1}, ..., x^{i,j-1}]
  - 例如 x^{0,2}：上采样 x^{1,1}，然后与 x^{0,0}, x^{0,1} 拼接后卷积
- 深度监督（deep_supervision）：
  - True: x^{0,1}, x^{0,2}, x^{0,3}, x^{0,4} 各接一个1x1卷积输出，训练时计算平均loss
  - False: 仅使用 x^{0,4} 的输出（简化模式）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DoubleConv, OutConv


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) 网络结构
    
    网络使用5个层级（L0-L4），通过嵌套的密集跳跃连接融合多尺度特征。
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 1, bilinear: bool = False, deep_supervision: bool = False):
        """
        初始化UNet++模型
        
        Args:
            in_channels: 输入图像通道数，灰度图为1，RGB图为3
            num_classes: 分割类别数，二分类为1，多分类按实际数量设置
            bilinear: 是否使用双线性插值上采样，False使用转置卷积
            deep_supervision: 是否使用深度监督，True则输出多个分支用于训练
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        
        # 基础通道数
        base_ch = 64
        
        # 通道数配置：5个层级（i=0到i=4）
        # bilinear=False（转置卷积）：[64, 128, 256, 512, 1024]
        # bilinear=True（双线性插值）：最后一级不翻倍，为[64, 128, 256, 512, 512]
        factor = 2 if bilinear else 1
        channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16 // factor]
        
        # -------------------- 编码器路径（j=0） --------------------
        # x^{0,0}: 输入层，对原始图像进行双卷积
        # 通道数：in_channels -> 64，空间尺寸不变
        self.conv00 = DoubleConv(in_channels, channels[0])
        
        # x^{1,0}: 第一次下采样
        # MaxPool2d(2) -> DoubleConv，空间尺寸减半，通道数翻倍：64 -> 128
        self.down10 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(channels[0], channels[1])
        )
        
        # x^{2,0}: 第二次下采样
        # 128 -> 256，空间尺寸变为1/4
        self.down20 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(channels[1], channels[2])
        )
        
        # x^{3,0}: 第三次下采样
        # 256 -> 512，空间尺寸变为1/8
        self.down30 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(channels[2], channels[3])
        )
        
        # x^{4,0}: 第四次下采样（瓶颈层）
        # 512 -> 1024(转置卷积) 或 512(双线性)，空间尺寸变为1/16
        self.down40 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(channels[3], channels[4])
        )
        
        # -------------------- 上采样模块（用于上采样 x^{i+1,j-1}） --------------------
        # 每个 x^{i,j}（j>0）都需要一个上采样模块将 x^{i+1,j-1} 上采样到 i 层分辨率
        # 上采样后通道数统一为 channels[i]
        # 使用 nn.ModuleDict 按索引存储，键格式为 'up{i}{j}'
        
        self.ups = nn.ModuleDict()
        
        # -------------------- 嵌套卷积模块（融合拼接后的特征） --------------------
        # 每个 x^{i,j}（j>0）都需要一个 DoubleConv 模块
        # 输入通道数 = 上采样后通道数(channels[i]) + 拼接的特征数(j) * channels[i]
        #            = (j + 1) * channels[i]
        # 输出通道数 = channels[i]
        
        self.convs = nn.ModuleDict()
        
        # 构建 j=1 层的节点：x^{0,1}, x^{1,1}, x^{2,1}, x^{3,1}
        for i in range(4):
            # 上采样模块：将 x^{i+1,0}（channels[i+1]通道）上采样到 i 层，输出 channels[i] 通道
            self.ups[f'up{i}1'] = self._make_upsample(channels[i+1], channels[i])
            # 卷积模块：输入是上采样特征(channels[i]) + x^{i,0}(channels[i]) = 2*channels[i]
            self.convs[f'conv{i}1'] = DoubleConv(2 * channels[i], channels[i])
        
        # 构建 j=2 层的节点：x^{0,2}, x^{1,2}, x^{2,2}
        for i in range(3):
            # 上采样模块：将 x^{i+1,1}（channels[i+1]通道）上采样到 i 层
            self.ups[f'up{i}2'] = self._make_upsample(channels[i+1], channels[i])
            # 卷积模块：输入是上采样特征(channels[i]) + x^{i,0} + x^{i,1} = 3*channels[i]
            self.convs[f'conv{i}2'] = DoubleConv(3 * channels[i], channels[i])
        
        # 构建 j=3 层的节点：x^{0,3}, x^{1,3}
        for i in range(2):
            # 上采样模块：将 x^{i+1,2}（channels[i+1]通道）上采样到 i 层
            self.ups[f'up{i}3'] = self._make_upsample(channels[i+1], channels[i])
            # 卷积模块：输入是上采样特征 + x^{i,0} + x^{i,1} + x^{i,2} = 4*channels[i]
            self.convs[f'conv{i}3'] = DoubleConv(4 * channels[i], channels[i])
        
        # 构建 j=4 层的节点：x^{0,4}
        # 上采样模块：将 x^{1,3}（channels[1]=128通道）上采样到 i=0 层
        self.ups['up04'] = self._make_upsample(channels[1], channels[0])
        # 卷积模块：输入是上采样特征 + x^{0,0} + x^{0,1} + x^{0,2} + x^{0,3} = 5*channels[i]
        self.convs['conv04'] = DoubleConv(5 * channels[0], channels[0])
        
        # -------------------- 输出层 --------------------
        if deep_supervision:
            # 深度监督模式：x^{0,1}, x^{0,2}, x^{0,3}, x^{0,4} 各接一个1x1卷积输出
            # 共4个输出分支，训练时计算平均loss
            self.out1 = OutConv(channels[0], num_classes)
            self.out2 = OutConv(channels[0], num_classes)
            self.out3 = OutConv(channels[0], num_classes)
            self.out4 = OutConv(channels[0], num_classes)
        else:
            # 非深度监督模式：仅使用 x^{0,4} 的输出
            self.outc = OutConv(channels[0], num_classes)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"UNet++模型参数量: {total_params:,}")
    
    def _make_upsample(self, in_ch: int, out_ch: int) -> nn.Module:
        """
        创建上采样模块
        
        Args:
            in_ch: 输入通道数（来自深层特征）
            out_ch: 输出通道数（上采样后应为浅层通道数）
        
        Returns:
            上采样模块：转置卷积或双线性插值+1x1卷积
        """
        if self.bilinear:
            # 双线性插值上采样：
            # 1. Upsample：空间尺寸翻倍，通道数不变（仍为in_ch）
            # 2. 1x1 Conv：将通道数从in_ch映射到out_ch
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            )
        else:
            # 转置卷积（反卷积）上采样：
            # kernel_size=2, stride=2：空间尺寸翻倍，通道数从in_ch变为out_ch
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    
    def _center_pad(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        将x居中填充到与target相同的空间尺寸
        
        Args:
            x: 需要填充的特征图
            target: 目标尺寸的特征图
        
        Returns:
            填充后的特征图
        """
        # 计算尺寸差异
        diffY = target.size()[2] - x.size()[2]  # 高度差
        diffX = target.size()[3] - x.size()[3]  # 宽度差
        
        # 使用F.pad居中填充
        # pad格式：[左, 右, 上, 下]
        x = F.pad(x, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        return x
    
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
            如果deep_supervision=True：返回logits列表，包含4个输出，每个形状 (B, num_classes, H, W)
            如果deep_supervision=False：返回单个logits，形状 (B, num_classes, H, W)
            注意：输出不经过sigmoid/softmax激活，由损失函数统一处理
        """
        # ==================== 编码器前向传播（j=0） ====================
        # x00: x^{0,0}，输入层特征
        # 形状：(B, in_channels, H, W) -> (B, 64, H, W)
        x00 = self.conv00(x)
        
        # x10: x^{1,0}，第一次下采样特征
        # 形状：(B, 64, H, W) -> (B, 128, H/2, W/2)
        x10 = self.down10(x00)
        
        # x20: x^{2,0}，第二次下采样特征
        # 形状：(B, 128, H/2, W/2) -> (B, 256, H/4, W/4)
        x20 = self.down20(x10)
        
        # x30: x^{3,0}，第三次下采样特征
        # 形状：(B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
        x30 = self.down30(x20)
        
        # x40: x^{4,0}，瓶颈层特征
        # 形状：(B, 512, H/8, W/8) -> (B, 1024或512, H/16, W/16)
        x40 = self.down40(x30)
        
        # ==================== 嵌套跳跃连接前向传播 ====================
        
        # ---------- j=1 层 ----------
        # x01: x^{0,1} = Conv( up(x^{1,0}) cat x^{0,0} )
        # 上采样 x10 到 x00 的尺寸，然后拼接卷积
        x01 = self.ups['up01'](x10)
        x01 = self._center_pad(x01, x00)
        x01 = torch.cat([x00, x01], dim=1)
        x01 = self.convs['conv01'](x01)
        
        # x11: x^{1,1} = Conv( up(x^{2,0}) cat x^{1,0} )
        x11 = self.ups['up11'](x20)
        x11 = self._center_pad(x11, x10)
        x11 = torch.cat([x10, x11], dim=1)
        x11 = self.convs['conv11'](x11)
        
        # x21: x^{2,1} = Conv( up(x^{3,0}) cat x^{2,0} )
        x21 = self.ups['up21'](x30)
        x21 = self._center_pad(x21, x20)
        x21 = torch.cat([x20, x21], dim=1)
        x21 = self.convs['conv21'](x21)
        
        # x31: x^{3,1} = Conv( up(x^{4,0}) cat x^{3,0} )
        x31 = self.ups['up31'](x40)
        x31 = self._center_pad(x31, x30)
        x31 = torch.cat([x30, x31], dim=1)
        x31 = self.convs['conv31'](x31)
        
        # ---------- j=2 层 ----------
        # x02: x^{0,2} = Conv( up(x^{1,1}) cat x^{0,0} cat x^{0,1} )
        x02 = self.ups['up02'](x11)
        x02 = self._center_pad(x02, x00)
        x02 = torch.cat([x00, x01, x02], dim=1)
        x02 = self.convs['conv02'](x02)
        
        # x12: x^{1,2} = Conv( up(x^{2,1}) cat x^{1,0} cat x^{1,1} )
        x12 = self.ups['up12'](x21)
        x12 = self._center_pad(x12, x10)
        x12 = torch.cat([x10, x11, x12], dim=1)
        x12 = self.convs['conv12'](x12)
        
        # x22: x^{2,2} = Conv( up(x^{3,1}) cat x^{2,0} cat x^{2,1} )
        x22 = self.ups['up22'](x31)
        x22 = self._center_pad(x22, x20)
        x22 = torch.cat([x20, x21, x22], dim=1)
        x22 = self.convs['conv22'](x22)
        
        # ---------- j=3 层 ----------
        # x03: x^{0,3} = Conv( up(x^{1,2}) cat x^{0,0} cat x^{0,1} cat x^{0,2} )
        x03 = self.ups['up03'](x12)
        x03 = self._center_pad(x03, x00)
        x03 = torch.cat([x00, x01, x02, x03], dim=1)
        x03 = self.convs['conv03'](x03)
        
        # x13: x^{1,3} = Conv( up(x^{2,2}) cat x^{1,0} cat x^{1,1} cat x^{1,2} )
        x13 = self.ups['up13'](x22)
        x13 = self._center_pad(x13, x10)
        x13 = torch.cat([x10, x11, x12, x13], dim=1)
        x13 = self.convs['conv13'](x13)
        
        # ---------- j=4 层 ----------
        # x04: x^{0,4} = Conv( up(x^{1,3}) cat x^{0,0} cat x^{0,1} cat x^{0,2} cat x^{0,3} )
        x04 = self.ups['up04'](x13)
        x04 = self._center_pad(x04, x00)
        x04 = torch.cat([x00, x01, x02, x03, x04], dim=1)
        x04 = self.convs['conv04'](x04)
        
        # ==================== 输出层 ====================
        if self.deep_supervision:
            # 深度监督模式：返回4个输出分支
            # 训练时对这4个输出分别计算loss然后取平均
            logits1 = self.out1(x01)
            logits2 = self.out2(x02)
            logits3 = self.out3(x03)
            logits4 = self.out4(x04)
            return [logits1, logits2, logits3, logits4]
        else:
            # 非深度监督模式：仅使用 x^{0,4} 的输出
            logits = self.outc(x04)
            return logits
