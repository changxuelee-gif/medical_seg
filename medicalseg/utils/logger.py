"""统一日志模块

提供项目统一的日志设置，同时输出到控制台和文件。
日志格式包含时间、级别、消息，支持按日期命名日志文件。
使用pathlib处理路径，跨平台兼容。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = 'medicalseg',
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """设置并返回logger对象

    创建一个同时输出到控制台和文件的logger。日志格式包含：
    - 时间戳（精确到毫秒）
    - 日志级别
    - 模块名称
    - 日志消息

    Args:
        name: logger名称，默认为'medicalseg'
        log_dir: 日志文件目录，默认为项目根目录下的logs/
        level: logger的根级别，默认为INFO
        console_level: 控制台输出级别，默认为INFO
        file_level: 文件输出级别，默认为DEBUG（记录所有详细信息）
        log_filename: 日志文件名，默认按当前日期生成（如：2024-01-15.log）

    Returns:
        配置好的logging.Logger对象

    Examples:
        >>> logger = setup_logger()
        >>> logger.info("训练开始")
        [2024-01-15 10:30:45,123] [INFO] [medicalseg] 训练开始
    """
    # 获取logger
    logger = logging.getLogger(name)

    # 如果logger已经配置过handler，直接返回，避免重复添加
    if logger.handlers:
        return logger

    # 设置logger根级别
    logger.setLevel(level)

    # 防止日志传播到父logger
    logger.propagate = False

    # 定义日志格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ========== 控制台处理器 ==========
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ========== 文件处理器 ==========
    if log_dir is None:
        # 默认日志目录：项目根目录/logs
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        log_dir = project_root / 'logs'

    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成日志文件名
    if log_filename is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_filename = f'{current_date}.log'

    log_file_path = log_dir / log_filename

    file_handler = logging.FileHandler(
        filename=log_file_path,
        encoding='utf-8',
        mode='a'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 输出初始化信息
    logger.debug(f"日志初始化完成，日志文件: {log_file_path}")

    return logger


def get_logger(name: str = 'medicalseg') -> logging.Logger:
    """获取已配置的logger

    如果logger尚未配置，会使用默认配置自动初始化。

    Args:
        name: logger名称，默认为'medicalseg'

    Returns:
        logging.Logger对象
    """
    logger = logging.getLogger(name)

    # 如果没有handler，使用默认配置初始化
    if not logger.handlers:
        logger = setup_logger(name=name)

    return logger


class LoggerWriter:
    """将logger伪装成文件对象的适配器类

    用于重定向stdout/stderr到日志系统，例如捕获print输出。
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """初始化LoggerWriter

        Args:
            logger: 要写入的logger对象
            level: 日志级别，默认为INFO
        """
        self.logger = logger
        self.level = level
        self._buffer = ''

    def write(self, message: str) -> None:
        """写入消息（文件对象接口）

        Args:
            message: 要写入的消息
        """
        # 处理换行缓冲
        if message.endswith('\n'):
            self._buffer += message[:-1]
            if self._buffer:
                self.logger.log(self.level, self._buffer)
            self._buffer = ''
        else:
            self._buffer += message

    def flush(self) -> None:
        """刷新缓冲区（文件对象接口）"""
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ''
