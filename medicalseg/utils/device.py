"""设备检测模块

自动检测CUDA是否可用，返回合适的torch.device对象。
支持打印详细的设备信息，方便调试和确认运行环境。
"""

from __future__ import annotations

import torch
from torch import backends, cuda


def get_device(use_cuda: bool = True, verbose: bool = True) -> torch.device:
    """获取可用的计算设备

    自动检测CUDA是否可用，如果可用且use_cuda为True则返回GPU设备，
    否则返回CPU设备。可选择打印详细设备信息。

    Args:
        use_cuda: 是否尝试使用CUDA，默认为True
        verbose: 是否打印设备信息，默认为True

    Returns:
        torch.device对象，表示计算设备（'cuda:0' 或 'cpu'）

    Examples:
        >>> device = get_device()
        使用设备: cuda:0 (NVIDIA GeForce RTX 3090)
        >>> device
        device(type='cuda', index=0)

        >>> device = get_device(use_cuda=False)
        使用设备: cpu
    """
    # 判断CUDA是否可用
    cuda_available = cuda.is_available() and use_cuda

    if cuda_available:
        device = torch.device('cuda:0')
        if verbose:
            gpu_name = cuda.get_device_name(0)
            gpu_count = cuda.device_count()
            cuda_version = torch.version.cuda
            cudnn_version = backends.cudnn.version() if backends.cudnn.is_available() else '不可用'

            print("=" * 60)
            print("计算设备信息")
            print("=" * 60)
            print(f"  设备类型: GPU (CUDA)")
            print(f"  设备名称: {gpu_name}")
            print(f"  CUDA设备数量: {gpu_count}")
            print(f"  CUDA版本: {cuda_version}")
            print(f"  cuDNN版本: {cudnn_version}")
            print(f"  当前设备: cuda:0")
            print("=" * 60)
    else:
        device = torch.device('cpu')
        if verbose:
            print("=" * 60)
            print("计算设备信息")
            print("=" * 60)
            print(f"  设备类型: CPU")
            if use_cuda and not cuda.is_available():
                print(f"  注意: CUDA不可用，已自动切换到CPU")
            print("=" * 60)

    return device


def get_device_count() -> int:
    """获取可用的CUDA设备数量

    Returns:
        CUDA设备数量，如果CUDA不可用则返回0
    """
    return cuda.device_count() if cuda.is_available() else 0


def print_gpu_memory_info(device_id: int = 0) -> None:
    """打印指定GPU的显存信息

    Args:
        device_id: GPU设备ID，默认为0
    """
    if not cuda.is_available():
        print("CUDA不可用，无法查询显存信息")
        return

    if device_id >= cuda.device_count():
        print(f"设备ID {device_id} 不存在，共有 {cuda.device_count()} 个GPU")
        return

    # 设置当前设备
    cuda.set_device(device_id)

    # 获取显存信息（单位：GB）
    total_mem = cuda.get_device_properties(device_id).total_mem / (1024 ** 3)
    allocated_mem = cuda.memory_allocated(device_id) / (1024 ** 3)
    reserved_mem = cuda.memory_reserved(device_id) / (1024 ** 3)
    free_mem = total_mem - reserved_mem

    print("-" * 60)
    print(f"GPU {device_id} 显存信息: {cuda.get_device_name(device_id)}")
    print("-" * 60)
    print(f"  总显存: {total_mem:.2f} GB")
    print(f"  已分配: {allocated_mem:.2f} GB")
    print(f"  已预留: {reserved_mem:.2f} GB")
    print(f"  可用: {free_mem:.2f} GB")
    print("-" * 60)
