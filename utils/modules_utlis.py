#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/9/22 10:16 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/22 10:16   wangfc      1.0         None
"""
import sys

def show_sys_path(search_dir=None):
    if search_dir:
        sys.path.append(search_dir)
    print("sys.path:\n")
    for s in sys.path:
        print(f"{s}")