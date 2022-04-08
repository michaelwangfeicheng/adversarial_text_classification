#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/5 16:57 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/5 16:57   wangfc      1.0         None
"""
import torch

class ModelConfig(object):
    """配置参数"""
    def __init__(self, model_name,adversarial_training_mode=None):
        self.model_name = model_name
        self.adversarial_training_mode = adversarial_training_mode


    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, torch.Tensor)}

    def __getitem__(self, item):
        return self.__getattribute__(item)