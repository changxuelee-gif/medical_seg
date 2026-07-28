#!/usr/bin/env python3
"""
医学图像分割推理入口脚本

使用方法：
    python tools/predict.py --input data/test/images/case001.png           # 单张图片推理
    python tools/predict.py --input data/test/images --output outputs/results  # 批量目录推理
    python tools/predict.py --input img.png --checkpoint checkpoints/best_model.pth --threshold 0.6
    python tools/predict.py --input img.png --no_overlay                   # 不保存叠加图

功能说明：
    1. 解析命令行参数
    2. 设置 sys.path 确保能正确导入 medicalseg 包
    3. 创建 Predictor 推理器实例
    4. 判断输入是文件还是目录
    5. 执行单文件推理或批量目录推理
    6. 保存可视化结果和预测掩码
"""

import argparse
import sys
from pathlib import Path

# ========== 设置项目根目录到 sys.path ==========
# 当前脚本位于 tools/predict.py，向上一级即为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medicalseg.inference import Predictor


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='医学图像分割推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 单张图片推理，使用默认参数
  python tools/predict.py --input data/test/images/case001.png
  
  # 批量目录推理，指定输出目录
  python tools/predict.py --input data/test/images --output outputs/predictions
  
  # 指定模型和阈值
  python tools/predict.py --input img.png --checkpoint checkpoints/best_model.pth --threshold 0.6
  
  # 不保存叠加图
  python tools/predict.py --input img.png --no_overlay
        """
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入图片路径或目录路径（必填）'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/predictions',
        help='输出结果保存目录，默认为 outputs/predictions'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='checkpoints/best_model.pth',
        help='模型检查点路径，默认为 checkpoints/best_model.pth'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='二值化阈值，范围[0,1]，默认0.5'
    )

    overlay_group = parser.add_mutually_exclusive_group()
    overlay_group.add_argument(
        '--save_overlay',
        action='store_true',
        dest='save_overlay',
        default=True,
        help='保存叠加图（默认开启）'
    )
    overlay_group.add_argument(
        '--no_overlay',
        action='store_false',
        dest='save_overlay',
        help='不保存叠加图'
    )

    return parser.parse_args()


def main():
    """推理主函数"""
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"错误: 模型检查点不存在: {checkpoint_path}")
        sys.exit(1)

    print("=" * 60)
    print("医学图像分割推理")
    print("=" * 60)
    print(f"  输入路径: {input_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  模型路径: {checkpoint_path}")
    print(f"  二值化阈值: {args.threshold}")
    print(f"  保存叠加图: {'是' if args.save_overlay else '否'}")
    print("=" * 60)

    print("\n正在加载模型...")
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        device=None,
        cfg=None
    )

    if input_path.is_file():
        print(f"\n正在对单张图片进行推理: {input_path.name}")
        original_image, pred_mask, prob_map = predictor.predict_file(
            image_path=input_path,
            save_dir=output_dir,
            save_overlay=args.save_overlay,
            threshold=args.threshold
        )
        print(f"推理完成，结果保存到: {output_dir}")
    elif input_path.is_dir():
        print(f"\n正在对目录进行批量推理: {input_path}")
        count = predictor.predict_dir(
            input_dir=input_path,
            save_dir=output_dir,
            pattern='*.png',
            save_overlay=args.save_overlay,
            threshold=args.threshold
        )
        if count == 0:
            print("未找到任何图片文件，请检查输入路径和文件格式")
            sys.exit(1)
    else:
        print(f"错误: 输入路径既不是文件也不是目录: {input_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"推理完成，结果保存到 {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
