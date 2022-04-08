#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@time: 2021/3/1 16:56

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/1 16:56   wangfc      1.0         None

"""


import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def cdroot():
    """
    cd to project root, so models are saved in the root folder
    """
    os.chdir(root)