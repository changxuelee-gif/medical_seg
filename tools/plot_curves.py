#!/usr/bin/env python3
"""
医学图像分割训练曲线绘制脚本

使用方法：
    python tools/plot_curves.py                                    # 使用默认路径
    python tools/plot_curves.py --metrics logs/metrics.json        # 指定 metrics 文件
    python tools/plot_curves.py --output outputs/figures           # 指定输出目录

功能说明：
    1. 解析命令行参数
    2. 设置 sys.path 确保能正确导入 medicalseg 包
    3. 加载训练历史 JSON 文件（metrics.json）
    4. 调用 plot_metric_curves 绘制 loss、dice、iou 三条主要曲线
    5. 打印保存路径信息
"""

import argparse
import json
import sys
from pathlib import Path

# ========== 设置项目根目录到 sys.path ==========
# 当前脚本位于 tools/plot_curves.py，向上一级即为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 现在可以导入 visualization 模块
from medicalseg.visualization.plotter import plot_metric_curves


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='医学图像分割训练曲线绘制工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python tools/plot_curves.py
  python tools/plot_curves.py --metrics logs/metrics.json
  python tools/plot_curves.py --output outputs/figures
  python tools/plot_curves.py --metrics checkpoints/metrics.json --output my_figs
        """
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='logs/metrics.json',
        help='训练指标 JSON 文件路径，默认为 logs/metrics.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/figures',
        help='图片输出目录，默认为 outputs/figures'
    )
    return parser.parse_args()


def main():
    """主函数：加载 metrics.json 并绘制训练曲线"""
    # ========== 1. 解析命令行参数 ==========
    args = parse_args()

    # ========== 2. 处理路径 ==========
    # metrics 文件路径：相对于项目根目录解析
    metrics_path = PROJECT_ROOT / args.metrics
    # 输出目录路径：相对于项目根目录解析
    output_dir = PROJECT_ROOT / args.output

    print("=" * 80)
    print("训练曲线绘制工具")
    print("=" * 80)

    # ========== 3. 检查 metrics 文件是否存在 ==========
    if not metrics_path.exists():
        print(f"[错误] 训练指标文件不存在: {metrics_path}")
        print("提示：请先运行训练脚本完成训练，例如: python tools/train.py")
        sys.exit(1)

    print(f"[信息] 正在加载训练指标文件: {metrics_path}")

    # ========== 4. 加载 JSON 文件 ==========
    with open(metrics_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    # 打印加载到的指标信息
    available_metrics = [k for k in history.keys() if len(history[k]) > 0]
    print(f"[信息] 加载成功，包含以下指标记录:")
    for metric in available_metrics:
        print(f"  - {metric}: {len(history[metric])} 个 epoch 记录")

    # ========== 5. 绘制曲线 ==========
    print("-" * 80)
    print(f"[信息] 正在绘制训练曲线，输出目录: {output_dir}")
    plot_metric_curves(history=history, save_dir=output_dir)

    # ========== 6. 完成 ==========
    print("=" * 80)
    print("绘制完成！生成的图片文件：")
    print(f"  1. 损失曲线: {output_dir / 'loss_curve.png'}")
    print(f"  2. Dice 曲线: {output_dir / 'dice_curve.png'}")
    print(f"  3. IoU 曲线:  {output_dir / 'iou_curve.png'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
