"""
医学图像分割训练曲线绘图模块

提供训练过程中损失和评估指标的可视化功能，包括：
- 单个指标曲线绘制（支持 train/val loss 对比）
- 批量绘制主要指标曲线（loss、dice、iou）
- matplotlib 中文字体配置
"""

# 重要：在导入 pyplot 之前设置后端为 Agg，确保无 GUI 环境（如服务器）下也能保存图片
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List


def setup_chinese_font() -> None:
    """
    配置 matplotlib 中文字体，解决中文显示乱码问题
    
    按优先级尝试多种常见中文字体：
    - SimHei: Windows 系统常用黑体
    - PingFang SC: macOS 系统苹方字体
    - Microsoft YaHei: Windows 微软雅黑
    - Heiti TC: macOS 黑体-繁
    - Arial Unicode MS: 跨平台通用字体
    - STHeiti: macOS 华文黑体
    
    同时设置 axes.unicode_minus = False 解决负号显示为方块的问题
    """
    # 定义要尝试的中文字体列表，按优先级排序
    chinese_fonts: List[str] = [
        'SimHei',           # Windows 黑体
        'PingFang SC',      # macOS 苹方
        'Microsoft YaHei',  # Windows 微软雅黑
        'Heiti TC',         # macOS 黑体-繁
        'Arial Unicode MS', # 跨平台通用
        'STHeiti',          # macOS 华文黑体
        'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
        'Noto Sans CJK SC', # Google Noto 中文字体
    ]

    # 从 matplotlib 的字体管理器中获取所有已安装字体名称
    from matplotlib import font_manager
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)

    # 按优先级查找第一个可用的中文字体
    selected_font = None
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    # 如果找到了可用字体，进行配置
    if selected_font is not None:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        print(f"[plotter] 已配置中文字体: {selected_font}")
    else:
        print("[plotter] 警告：未找到可用的中文字体，中文标签可能显示为方块")

    # 解决负号显示问题：默认情况下 unicode 负号可能无法在中文字体下正常显示
    plt.rcParams['axes.unicode_minus'] = False


def plot_single_curve(
    history: Dict[str, List[float]],
    metric_name: str,
    title: str,
    save_path: Path,
    ylabel: Optional[str] = None
) -> None:
    """
    绘制单个指标的训练曲线
    
    根据 metric_name 的不同，绘制不同的曲线：
    - 'loss': 同时绘制 train_loss 和 val_loss 两条曲线，便于对比训练和验证损失
    - 其他指标（'dice', 'iou', 'precision', 'recall', 'accuracy'）: 只绘制 val_xxx 曲线
    
    Args:
        history: 训练历史字典，包含 'train_loss', 'val_loss', 'val_dice' 等键，
                 每个键对应一个列表，记录每个 epoch 的指标值
        metric_name: 要绘制的指标名称，支持 'loss', 'dice', 'iou', 'precision', 'recall', 'accuracy'
        title: 图表标题
        save_path: 图片保存路径（Path 对象）
        ylabel: Y 轴标签，若为 None 则使用 metric_name
    """
    # 确保保存路径的父目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建新的图形，设置合适的尺寸和分辨率
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # 生成 epoch 序列（从 1 开始，更符合人类习惯）
    epochs = range(1, len(history.get('train_loss', history.get(f'val_{metric_name}', []))) + 1)

    if metric_name == 'loss':
        # 损失曲线：同时绘制训练损失和验证损失
        if 'train_loss' in history:
            ax.plot(
                epochs,
                history['train_loss'],
                label='训练损失 (Train Loss)',
                color='#2196F3',   # 蓝色：训练
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8
            )
        if 'val_loss' in history:
            ax.plot(
                epochs,
                history['val_loss'],
                label='验证损失 (Val Loss)',
                color='#F44336',   # 红色：验证
                linewidth=2,
                marker='s',
                markersize=3,
                alpha=0.8
            )
    else:
        # 其他指标：只绘制验证集指标（因为训练集通常不计算这些指标）
        val_key = f'val_{metric_name}'
        if val_key in history:
            # 为不同指标选择美观的颜色
            color_map = {
                'dice': '#4CAF50',       # 绿色：Dice
                'iou': '#FF9800',        # 橙色：IoU
                'precision': '#9C27B0',  # 紫色：精确率
                'recall': '#00BCD4',     # 青色：召回率
                'accuracy': '#795548',   # 棕色：准确率
            }
            line_color = color_map.get(metric_name, '#3F51B5')  # 默认靛蓝色

            # 指标中文名称映射
            metric_labels = {
                'dice': 'Dice 系数',
                'iou': 'IoU 交并比',
                'precision': '精确率 (Precision)',
                'recall': '召回率 (Recall)',
                'accuracy': '准确率 (Accuracy)',
            }
            label = metric_labels.get(metric_name, f'Val {metric_name}')

            ax.plot(
                epochs,
                history[val_key],
                label=label,
                color=line_color,
                linewidth=2,
                marker='o',
                markersize=4,
                alpha=0.8
            )

    # 设置 X 轴标签
    ax.set_xlabel('Epoch (训练轮次)', fontsize=12, fontweight='bold')

    # 设置 Y 轴标签：优先使用传入的 ylabel，否则使用 metric_name
    if ylabel is None:
        if metric_name == 'loss':
            ylabel = '损失值 (Loss)'
        else:
            metric_ylabels = {
                'dice': 'Dice 系数',
                'iou': 'IoU 交并比',
                'precision': '精确率',
                'recall': '召回率',
                'accuracy': '准确率',
            }
            ylabel = metric_ylabels.get(metric_name, metric_name)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # 设置图表标题
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # 添加图例，设置合适的位置和字体大小
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # 添加网格线，便于读数
    ax.grid(True, linestyle='--', alpha=0.6)

    # 设置 X 轴刻度为整数（epoch 是整数）
    ax.set_xticks(list(epochs))
    # 如果 epoch 较多，旋转刻度标签避免重叠
    if len(epochs) > 20:
        plt.xticks(rotation=45)

    # 调整布局，防止标签被截断
    plt.tight_layout()

    # 保存图片到指定路径
    # dpi=150 保证图片清晰度，bbox_inches='tight' 自动裁剪多余白边
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，释放内存


def plot_metric_curves(
    history: Dict[str, List[float]],
    save_dir: Path
) -> None:
    """
    批量绘制三个主要指标曲线：loss、dice、iou
    
    这三个是医学图像分割中最核心的监控指标：
    - loss_curve.png: 训练/验证损失曲线，判断是否过拟合/欠拟合
    - dice_curve.png: Dice 系数曲线，最常用的分割评估指标
    - iou_curve.png: IoU 交并比曲线，另一个重要的分割指标
    
    Args:
        history: 训练历史字典，格式同 plot_single_curve
        save_dir: 图片保存目录（Path 对象），会自动创建（parents=True, exist_ok=True）
    """
    # 使用 pathlib 创建保存目录（包括所有父目录），已存在则不报错
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========== 绘制损失曲线 ==========
    loss_save_path = save_dir / 'loss_curve.png'
    plot_single_curve(
        history=history,
        metric_name='loss',
        title='训练与验证损失曲线',
        save_path=loss_save_path,
        ylabel='损失值 (Loss)'
    )
    print(f"[plotter] 损失曲线已保存至: {loss_save_path}")

    # ========== 绘制 Dice 系数曲线 ==========
    dice_save_path = save_dir / 'dice_curve.png'
    plot_single_curve(
        history=history,
        metric_name='dice',
        title='验证集 Dice 系数曲线',
        save_path=dice_save_path,
        ylabel='Dice 系数'
    )
    print(f"[plotter] Dice 系数曲线已保存至: {dice_save_path}")

    # ========== 绘制 IoU 曲线 ==========
    iou_save_path = save_dir / 'iou_curve.png'
    plot_single_curve(
        history=history,
        metric_name='iou',
        title='验证集 IoU 交并比曲线',
        save_path=iou_save_path,
        ylabel='IoU 交并比'
    )
    print(f"[plotter] IoU 曲线已保存至: {iou_save_path}")

    print(f"[plotter] 所有训练曲线已保存至目录: {save_dir}")


# 在模块导入时自动配置中文字体
setup_chinese_font()
