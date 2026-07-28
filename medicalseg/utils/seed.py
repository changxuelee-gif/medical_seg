"""随机种子固定模块

固定所有相关库的随机种子，确保实验结果的可复现性。
涵盖：Python random、NumPy、PyTorch（CPU/CUDA）、cuDNN等。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import backends, cuda


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """固定所有随机种子，保证实验可复现性

    统一设置以下随机数生成器的种子：
    - Python标准库random
    - NumPy随机数
    - PyTorch CPU随机数
    - PyTorch所有CUDA设备的随机数
    - 设置PYTHONHASHSEED环境变量

    同时配置cuDNN的确定性模式和基准模式。

    Args:
        seed: 随机种子值，默认为42
        deterministic: 是否启用cuDNN确定性模式，默认为True
                       启用后可保证结果复现，但可能降低性能
        benchmark: 是否启用cuDNN基准模式，默认为False
                   启用后会自动寻找最优算法，但会引入随机性

    Note:
        关于deterministic和benchmark的说明：
        - 追求完全可复现：deterministic=True, benchmark=False
        - 追求性能（不要求严格复现）：deterministic=False, benchmark=True
        - 完全可复现可能会带来一定的性能损失

    Examples:
        >>> set_seed(42)
        随机种子已固定为: 42
        配置: deterministic=True, benchmark=False
    """
    # 设置Python哈希种子（影响字典、集合等的迭代顺序）
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置Python random模块种子
    random.seed(seed)

    # 设置NumPy随机种子
    np.random.seed(seed)

    # 设置PyTorch CPU随机种子
    torch.manual_seed(seed)

    # 设置PyTorch所有CUDA设备的随机种子
    if cuda.is_available():
        cuda.manual_seed(seed)
        cuda.manual_seed_all(seed)

    # 配置cuDNN
    if deterministic:
        # 启用确定性模式，保证卷积等操作使用确定性算法
        backends.cudnn.deterministic = True
        # 禁用基准模式，防止自动选择最快的非确定性算法
        backends.cudnn.benchmark = False
    else:
        backends.cudnn.deterministic = False
        backends.cudnn.benchmark = benchmark

    print("=" * 60)
    print("随机种子已固定")
    print("=" * 60)
    print(f"  种子值: {seed}")
    print(f"  cuDNN确定性模式: {deterministic}")
    print(f"  cuDNN基准模式: {benchmark}")
    print(f"  PYTHONHASHSEED: {seed}")
    print("=" * 60)


def get_seed_state() -> dict:
    """获取当前所有随机数生成器的状态

    用于保存随机状态，方便后续恢复。

    Returns:
        包含各随机数生成器状态的字典
    """
    state = {
        'random_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(),
    }

    if cuda.is_available():
        state['torch_cuda_state'] = cuda.get_rng_state_all()

    return state


def set_seed_state(state: dict) -> None:
    """恢复随机数生成器状态

    从之前保存的状态字典恢复随机状态。

    Args:
        state: get_seed_state()返回的状态字典
    """
    random.setstate(state['random_state'])
    np.random.set_state(state['numpy_state'])
    torch.set_rng_state(state['torch_state'])

    if 'torch_cuda_state' in state and cuda.is_available():
        cuda.set_rng_state_all(state['torch_cuda_state'])
