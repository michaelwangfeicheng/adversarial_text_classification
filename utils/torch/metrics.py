#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/16 22:28 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/16 22:28   wangfc      1.0         None
"""

import torch

def binary_accuracy(output,label):
    # output = [batch_size, 2]
    # y_pred = [batch_size,2]
    output_softmax = torch.softmax(output,dim=-1)
    y_pred = torch.argmax(output,dim=-1,keepdim=False)
    correct = (y_pred==label).float()
    accuracy = correct.sum()/len(correct)
    return accuracy
