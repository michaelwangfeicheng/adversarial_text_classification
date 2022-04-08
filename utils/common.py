#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 15:45 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 15:45   wangfc      1.0         None
"""
import importlib
import inspect
import os
import logging
from collections import Callable
from types import TracebackType
from typing import Any, Optional, Text, Type
import time
from pathlib import Path

import shutil

from constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from utils.io import raise_deprecation_warning
from utils.time import TIME_FORMAT, LOG_DATE_FORMAT
from utils.constants import ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL, LOG_FORMAT


def set_log_level(log_level: Optional[int] = None) -> None:
    """Set log level of Rasa and Tensorflow either to the provided log level or
    to the log level specified in the environment variable 'LOG_LEVEL'. If none is set
    a default log level will be used."""

    if not log_level:
        log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level)

    logging.getLogger("faq").setLevel(log_level)

    # update_tensorflow_log_level()
    # update_asyncio_log_level()
    # update_matplotlib_log_level()
    # update_apscheduler_log_level()
    # update_socketio_log_level()

    os.environ[ENV_LOG_LEVEL] = logging.getLevelName(log_level)


def init_logger(output_dir=None, log_filename=None, log_file_path=None,
                log_level=DEFAULT_LOG_LEVEL,
                fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_filename, Path):
        log_filename = str(log_filename)
    if log_level is None:
        log_level =DEFAULT_LOG_LEVEL

    if isinstance(log_level, str):
        if log_level.lower() == 'debug':
            log_level = logging.DEBUG
        elif log_level.lower() == 'info':
            log_level = logging.INFO
        elif log_level.lower() == 'warn':
            log_level = logging.WARNING


    if output_dir and log_filename:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime(TIME_FORMAT, time.localtime())
        log_filename = f'{log_filename}_{timestamp}.log'
        log_file_path = os.path.join(output_dir, log_filename)

    log_format = logging.Formatter(fmt=fmt,
                                   datefmt=datefmt)

    # 当没有参数时, 默认访问`root logger`
    # 当使用 name的时候，初始化 faq logger
    logger = logging.getLogger()
    # logger = logging.getLogger('faq')

    # 增加 console_handler:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    # 增加 file_handler
    if log_file_path and log_file_path != '':
        # os.makedirs(os.path.dirname(log_file),exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)



    logger.setLevel(log_level)

    return logger


def is_logging_disabled() -> bool:
    """Returns `True` if log level is set to WARNING or ERROR, `False` otherwise."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    return log_level in ("ERROR", "WARNING")


def lazy_property(function: Callable) -> Any:
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = "_lazy_" + function.__name__

    @property
    def _lazyprop(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return _lazyprop



# from rasa.shared.utils.common import class_from_module_path

def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Any:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects.

    Args:
        module_path: either an absolute path to a Python class,
                     or the name of the class in the local / global scope.
        lookup_path: a path where to load the class from, if it cannot
                     be found in the local / global scope.

    Returns:
        a Python class

    Raises:
        ImportError, in case the Python class cannot be found.
    """
    klass = None
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        klass = getattr(m, class_name, None)
    elif lookup_path:
        # try to import the class from the lookup path
        m = importlib.import_module(lookup_path)
        klass = getattr(m, module_path, None)

    if klass is None:
        raise ImportError(f"Cannot retrieve class from path {module_path}.")

    if not inspect.isclass(klass):
        raise_deprecation_warning(
            f"`class_from_module_path()` is expected to return a class, "
            f"but {module_path} is not one. "
            f"This warning will be converted "
            f"into an exception in {NEXT_MAJOR_VERSION_FOR_DEPRECATIONS}."
        )

    return klass


# from rasa.nlu.utils import module_path_from_object
def module_path_from_object(o: Any) -> Text:
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__



class TempDirectoryPath(str):
    """Represents a path to an temporary directory. When used as a context
    manager, it erases the contents of the directory on exit.
    继承 str 类，生成一个 context manager,在退出 context 的时候，
    自动执行 __exit__() 删除临时目录，可以做到资源的释放

    """

    def __enter__(self) -> "TempDirectoryPath":
        return self

    def __exit__(
        self,
        _exc: Optional[Type[BaseException]],
        _value: Optional[Exception],
        _tb: Optional[TracebackType],
    ) -> bool:
        if os.path.exists(self):
            shutil.rmtree(self)
