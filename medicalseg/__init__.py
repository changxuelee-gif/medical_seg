"""medicalseg - 医疗影像分割工具包

提供医疗影像分割任务的完整解决方案，包括数据处理、模型构建、训练、推理和可视化。
"""

__version__ = '0.1.0'
__author__ = 'medicalseg team'

from . import utils
from .utils import (
    Config,
    load_config,
    load_default_config,
    setup_logger,
    get_logger,
)

try:
    from .utils import get_device, set_seed
    _torch_available = True
except ImportError:
    _torch_available = False
    get_device = None
    set_seed = None

__all__ = [
    '__version__',
    'utils',
    'Config',
    'load_config',
    'load_default_config',
    'setup_logger',
    'get_logger',
    'get_device',
    'set_seed',
]
