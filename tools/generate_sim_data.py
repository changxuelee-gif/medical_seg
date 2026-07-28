"""
模拟医疗影像数据生成工具

使用说明：
    在项目根目录下运行：
        python tools/generate_sim_data.py
    
    自定义参数示例：
        python tools/generate_sim_data.py --num_samples 200 --img_size 512 --output_dir data/raw/simulated
    
    参数说明：
        --num_samples  生成样本数量，默认100
        --img_size     图像尺寸（正方形），默认256
        --output_dir   输出根目录，默认data/raw/simulated

生成数据结构：
    output_dir/
    ├── images/      # 原始模拟影像PNG（灰度）
    ├── masks/       # 对应病灶掩码PNG（二值：0背景，255病灶）
    └── splits.txt   # 训练/验证/测试集划分文件（7:2:1）

模拟影像特点：
    - 模拟X光/CT灰度组织背景（灰度值50-100）
    - 添加高斯噪声模拟真实影像噪声
    - 包含1-3个圆形/椭圆形病灶（高亮区域，比背景高50-100灰度值）
    - 病灶边缘轻微模糊模拟部分容积效应
    - 整体轻微模糊模拟影像平滑效果
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ========== 设置项目根目录到 sys.path ==========
# 当前脚本位于 tools/generate_sim_data.py，向上一级即为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="生成模拟医疗影像分割数据")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="生成样本数量，默认100"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="图像尺寸（正方形），默认256"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/simulated",
        help="输出根目录，默认data/raw/simulated"
    )
    return parser.parse_args()


def generate_single_sample(img_size: int, seed: int = None):
    """
    生成单张模拟医疗影像及其对应掩码
    
    Args:
        img_size: 图像尺寸（宽高相同）
        seed: 随机种子，用于可复现性，默认None
    
    Returns:
        (image, mask): 图像uint8数组(H,W)和掩码uint8数组(H,W)，值域0-255
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    margin = 30
    background_gray = random.randint(50, 100)

    image = np.full((img_size, img_size), background_gray, dtype=np.float64)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    num_lesions = random.randint(1, 3)

    for _ in range(num_lesions):
        center_x = random.randint(margin, img_size - margin)
        center_y = random.randint(margin, img_size - margin)

        radius_a = random.randint(20, 60)
        radius_b = random.randint(20, 60)
        angle = random.uniform(0, 360)

        lesion_intensity = random.randint(50, 100)

        Y, X = np.ogrid[:img_size, :img_size]
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        x_rot = (X - center_x) * cos_angle + (Y - center_y) * sin_angle
        y_rot = -(X - center_x) * sin_angle + (Y - center_y) * cos_angle
        ellipse_mask = ((x_rot ** 2) / (radius_a ** 2) + (y_rot ** 2) / (radius_b ** 2)) <= 1.0

        lesion_layer = np.zeros_like(image)
        lesion_layer[ellipse_mask] = lesion_intensity

        edge_sigma = random.uniform(2, 5)
        kernel_size = int(edge_sigma * 6) | 1
        lesion_layer = cv2.GaussianBlur(lesion_layer, (kernel_size, kernel_size), edge_sigma)

        image += lesion_layer
        mask[ellipse_mask] = 255

    noise_std = random.uniform(3, 8)
    noise = np.random.normal(0, noise_std, (img_size, img_size))
    image += noise

    global_sigma = random.uniform(0.5, 1.0)
    kernel_size = int(global_sigma * 6) | 1
    if kernel_size < 3:
        kernel_size = 3
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), global_sigma)

    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, mask


def save_splits(train_files, val_files, test_files, output_dir: Path):
    """
    保存数据集划分信息到splits.txt文件
    
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        test_files: 测试集文件列表
        output_dir: 输出目录路径
    """
    splits_path = output_dir / "splits.txt"
    with open(splits_path, "w", encoding="utf-8") as f:
        f.write("# 数据集划分文件（训练:验证:测试 = 7:2:1）\n")
        f.write("# 格式：[train/val/test] 文件名\n\n")

        f.write(f"[train] ({len(train_files)} samples)\n")
        for img_path, _ in train_files:
            f.write(f"{img_path.name}\n")

        f.write(f"\n[val] ({len(val_files)} samples)\n")
        for img_path, _ in val_files:
            f.write(f"{img_path.name}\n")

        f.write(f"\n[test] ({len(test_files)} samples)\n")
        for img_path, _ in test_files:
            f.write(f"{img_path.name}\n")

    print(f"数据集划分已保存到: {splits_path.resolve()}")
    print(f"  训练集: {len(train_files)} 张")
    print(f"  验证集: {len(val_files)} 张")
    print(f"  测试集: {len(test_files)} 张")


def main():
    """主函数：生成模拟医疗影像数据集"""
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"模拟医疗影像数据生成")
    print(f"=" * 60)
    print(f"样本数量: {args.num_samples}")
    print(f"图像尺寸: {args.img_size}x{args.img_size}")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"图像保存: {images_dir.resolve()}")
    print(f"掩码保存: {masks_dir.resolve()}")
    print(f"-" * 60)

    num_digits = len(str(args.num_samples))

    for i in range(args.num_samples):
        image, mask = generate_single_sample(args.img_size)

        filename = f"sim_{i+1:0{num_digits}d}.png"
        img_path = images_dir / filename
        mask_path = masks_dir / filename

        cv2.imwrite(str(img_path), image)
        cv2.imwrite(str(mask_path), mask)

        if (i + 1) % 10 == 0 or (i + 1) == args.num_samples:
            print(f"已生成: {i+1}/{args.num_samples} 张  ({filename})")

    print(f"-" * 60)
    print(f"图像和掩码生成完成！")
    print(f"=" * 60)

    try:
        from medicalseg.datasets import train_val_test_split

        train_files, val_files, test_files = train_val_test_split(
            image_dir=images_dir,
            mask_dir=masks_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42,
        )
        save_splits(train_files, val_files, test_files, output_dir)
    except ImportError as e:
        print(f"注意: 无法导入medicalseg.datasets.train_val_test_split ({e})")
        print("数据集划分将由训练脚本自行完成。")
    except Exception as e:
        print(f"数据集划分时出错: {e}")
        print("数据集划分将由训练脚本自行完成。")

    print(f"\n全部完成！")


if __name__ == "__main__":
    main()
