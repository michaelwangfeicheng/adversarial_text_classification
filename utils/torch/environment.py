#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/8/26 17:19 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/26 17:19   wangfc      1.0         None
"""
import random
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)


def _reset_random_seed(seed=1234, n_gpu=0, ):
    logger.info('=' * 20 + 'Reset Random Seed to {}'.format(seed) + '=' * 20)
    # set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    # 保证每次结果一样
    torch.backends.cudnn.deterministic = True


def _setup_torch_environment(cuda_no=0, local_rank=-1, parallel_decorate=False, fp16=False):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # set device
    if local_rank == -1 or cuda_no:
        n_gpu = torch.cuda.device_count()
        if n_gpu>0 and parallel_decorate == False and cuda_no is not None:  # isinstance(self.no_cuda, str):
            # 单机单卡 device
            device = torch.device(f"cuda:{cuda_no}" if torch.cuda.is_available() else "cpu")
        else:
            # 单机多卡 device: 如果设备序号不存在，则为当前设备，即 torch.cuda.current_device() 的返回结果。
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            logger.info('使用设备 device={}'.format(device))

    else:
        # 还可以通过设备类型加上编号，来创建 device 对象：
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        if fp16:
            logger.info("16-bits training currently not supported in distributed training")
            fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device {} n_gpu {}".format(device, n_gpu))
    return device, n_gpu


def load_model_to_device(model:torch.nn.Module,device=None,gpu_no=0):
    if device is None:
        device = _setup_torch_environment(cuda_no=gpu_no)
    model.to(device=device)
    return model
