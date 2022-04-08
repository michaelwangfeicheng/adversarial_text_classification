#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/12/30 17:41 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/30 17:41   wangfc      1.0         None
"""

from functools import partial


def basic_tokenizer(ues_word=False):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    return tokenizer