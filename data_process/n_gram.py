#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/5 15:43 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/5 15:43   wangfc      1.0         None
"""

from typing import List

def generate_n_gram(x:List[str],n:int=2):
    n_grams = set((zip(*[x[i:] for i in range(n)])))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

if __name__ == '__main__':
    x = ['This', 'film', 'is', 'terrible']
    y = [x[i:] for i in range(3)]
    print(y)
    print(set(zip(*y)))