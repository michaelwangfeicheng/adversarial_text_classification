#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/17 16:07 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 16:07   wangfc      1.0         None
"""
import time
import datetime

import functools
from threading import current_thread
import os

TIME_FORMAT = '%Y%m%d_%H%M%S'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def get_current_time(fmt=TIME_FORMAT,with_time_stamp=False):
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    current_time = time.strftime(fmt,local_time)

    # 打印按指定格式排版的时间
    # time2 = datetime.datetime.now().strftime(fmt)
    if with_time_stamp:
        return current_time,time_stamp
    else:
        return current_time


def timeit(func):
    """
    @author:wangfc27441
    @desc:  对程序计时的装饰器
    @version：
    @time:2021/2/7 10:01

    Parameters
    ----------

    Returns
    -------
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        current_time,start_time = get_current_time(with_time_stamp=True)
        print(f'pid:{os.getpid()} thread_name:{current_thread().getName()} {func.__name__} 开始运行 at:{current_time}')
        result = func(*args,**kwargs)
        current_time,end_time = get_current_time(with_time_stamp=True)
        print(f'pid:{os.getpid()} thread_name:{current_thread().getName()} {func.__name__} 结束运行 at:{current_time}，共耗时 {end_time- start_time} secs')
        return result
    return wrapper