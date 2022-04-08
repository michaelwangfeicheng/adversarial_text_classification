#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/6 14:59 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/6 14:59   wangfc      1.0         None
"""
import torch

# from tutorial.fast_adversarial.CIFAR10.utils_fast_adversarial import clamp
# from tutorial.adversarial_text.adversarial_losses import _scale_l2

def clamp(X, lower_limit=0, upper_limit=1):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def l2_normalize(x,norm_length):
    """
    改版自 tf 版本

    """
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    # alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    # l2_norm = alpha * tf.sqrt(
    #     tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    # x_unit = x / l2_norm
    # return norm_length * x_unit

    # 沿着 dim 取绝对值和最大值
    batch_size ,seq_len,embedding_dim = x.shape
    # 取绝对值并 resize：(batch_size ,1)
    abs_x = torch.abs(x).resize_(batch_size,seq_len*embedding_dim)
    # alpha = (batch_size ,1)
    alpha, max_indices = torch.max(abs_x,dim=-1,keepdim=True)
    alpha = alpha+ 1e-12
    #  # alpha = (batch_size ,1,1)
    alpha.unsqueeze_(dim=-1)
    #  平方后resize :(batch_size ,seq_len*embedding_dim) -> (batch_size,1)
    powed = torch.pow(x / alpha, 2).resize_(batch_size,seq_len*embedding_dim)
    # l2_norm: (batch_size ,1)
    l2_norm  = alpha * torch.sqrt(torch.sum(powed,dim=-1,keepdim=True) + 1e-6).unsqueeze_(dim=-1)
    x_unit = x/l2_norm
    return norm_length * x_unit

