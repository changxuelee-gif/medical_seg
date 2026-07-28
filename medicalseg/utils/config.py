"""YAML配置加载模块

提供灵活的配置管理功能，支持：
- 加载YAML格式配置文件
- 默认配置与自定义配置深度合并
- 支持字典式属性访问（点号访问）
- 跨平台路径处理（使用pathlib）
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config(dict):
    """配置类，支持字典访问和点号属性访问

    继承自dict，可以像普通字典一样使用，同时支持通过点号访问属性。
    嵌套字典会自动转换为Config对象，实现层级化的属性访问。

    Examples:
        >>> cfg = Config({'a': {'b': 1}, 'c': 2})
        >>> cfg.a.b
        1
        >>> cfg['c']
        2
        >>> cfg.a.b = 3
        >>> cfg.a.b
        3
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """初始化Config对象

        将传入的字典或键值对转换为Config对象，嵌套字典会递归转换。

        Args:
            *args: 位置参数，传递给dict初始化
            **kwargs: 关键字参数，传递给dict初始化
        """
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, name: str) -> Any:
        """通过点号访问属性

        Args:
            name: 属性名称

        Returns:
            属性值

        Raises:
            AttributeError: 属性不存在时抛出
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """通过点号设置属性

        如果值是字典，会自动转换为Config对象。

        Args:
            name: 属性名称
            value: 属性值
        """
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[name] = value

    def __deepcopy__(self, memo: Dict[int, Any]) -> Config:
        """深拷贝支持

        Args:
            memo: 备忘录字典，用于处理循环引用

        Returns:
            深拷贝后的Config对象
        """
        return Config(copy.deepcopy(dict(self), memo))

    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典（递归转换所有嵌套Config）

        Returns:
            普通Python字典
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = copy.deepcopy(value)
        return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典

    递归合并override字典到base字典中：
    - 对于嵌套字典，递归合并
    - 对于其他类型，override的值会覆盖base的值

    Args:
        base: 基础字典（默认配置）
        override: 覆盖字典（自定义配置）

    Returns:
        合并后的新字典
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """加载YAML配置文件

    从指定路径加载YAML文件并解析为字典。

    Args:
        config_path: YAML配置文件路径，可以是字符串或Path对象

    Returns:
        解析后的配置字典

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
        yaml.YAMLError: YAML解析错误时抛出
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    default_config_path: Optional[Union[str, Path]] = None,
) -> Config:
    """加载配置，支持默认配置与自定义配置合并

    加载流程：
    1. 如果提供了default_config_path，先加载默认配置
    2. 如果提供了config_path，加载自定义配置并与默认配置深度合并
    3. 将结果转换为Config对象返回

    Args:
        config_path: 自定义配置文件路径（可选）
        default_config_path: 默认配置文件路径（可选）

    Returns:
        合并后的Config配置对象
    """
    base_config: Dict[str, Any] = {}

    # 加载默认配置
    if default_config_path is not None:
        base_config = load_yaml(default_config_path)

    # 加载并合并自定义配置
    if config_path is not None:
        custom_config = load_yaml(config_path)
        base_config = _deep_merge(base_config, custom_config)

    return Config(base_config)


def get_default_config_path() -> Path:
    """获取默认配置文件路径

    默认配置文件位于项目根目录下的 configs/default.yaml。
    通过当前文件位置向上查找项目根目录。

    Returns:
        默认配置文件的绝对路径
    """
    # 当前文件: medicalseg/utils/config.py
    # 向上两级到项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    default_path = project_root / 'configs' / 'default.yaml'
    return default_path


def load_default_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """加载默认配置并可选合并自定义配置

    这是一个便捷函数，自动查找并加载configs/default.yaml作为默认配置，
    然后可选地与用户提供的配置文件合并。

    Args:
        config_path: 自定义配置文件路径（可选），如提供则与默认配置合并

    Returns:
        合并后的Config配置对象
    """
    default_path = get_default_config_path()
    return load_config(config_path=config_path, default_config_path=default_path)
