"""medicalseg.utils 工具模块

提供项目通用的工具函数和类，包括：
- config: 配置加载与管理
- device: 计算设备检测（需要torch）
- logger: 统一日志设置
- seed: 随机种子固定（需要torch）
"""

from .config import Config, load_config, load_default_config, load_yaml, get_default_config_path
from .logger import setup_logger, get_logger, LoggerWriter

try:
    from .device import get_device, get_device_count, print_gpu_memory_info
    _torch_available = True
except ImportError:
    _torch_available = False
    get_device = None
    get_device_count = None
    print_gpu_memory_info = None

try:
    from .seed import set_seed, get_seed_state, set_seed_state
    if not _torch_available:
        set_seed = None
        get_seed_state = None
        set_seed_state = None
except ImportError:
    set_seed = None
    get_seed_state = None
    set_seed_state = None

__all__ = [
    # config
    'Config',
    'load_config',
    'load_default_config',
    'load_yaml',
    'get_default_config_path',
    # logger
    'setup_logger',
    'get_logger',
    'LoggerWriter',
    # device (require torch)
    'get_device',
    'get_device_count',
    'print_gpu_memory_info',
    # seed (require torch)
    'set_seed',
    'get_seed_state',
    'set_seed_state',
]
