#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/4 11:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/4 11:03   wangfc      1.0         None
"""
from typing import Dict, Text

from torch.nn import modules
from torch.nn.modules import Embedding
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch



class NormalizedEmbedding(Embedding):
    """
    参考： from tutorial.adversarial_text.layers import Embedding
    构建根据 vocab_freq 来进行 normalized的 Embedding layer
    embedding = (vocab_size, embedding_dim)
    1. 沿着 vocab 轴 计算每个维度的 mean,std
    2. 沿着 vocab 轴 使用 mean-std 对每个维度进行规范化
    """

    def __init__(self,normalize=True,vocab_freq:Dict[Text,int] = None, keep_prob=1, *args, **kwargs):
        super(NormalizedEmbedding,self).__init__(*args, **kwargs)
        self.normalize = normalize
        self.keep_prob = keep_prob
        self.vocab_freq = vocab_freq

        if normalize:
            assert vocab_freq is not None
            self.vocab_freq = torch.tensor(
                list(vocab_freq.values()), dtype=torch.float32) # shape=(vocab_size, 1))
            self.weight = self._normalize(self.weight)

    def _normalize(self, emb):
        # 计算 每个 word/char 的 freq_weight
        weights = self.vocab_freq / torch.sum(self.vocab_freq)
        if weights.dim()==1:
            weights = torch.unsqueeze(weights,dim=-1)
        # 沿着word/char 轴求平均 (vocab_size,1) * (vocab_size,embedding_dim) -> (vocab_size,embedding_dim) -> (1,embedding_dim)
        mean = torch.sum(weights * emb, 0, keepdim=True)
        # 沿着word/char 轴求 deviance:
        # (vocab_size,embedding_dim) - (1,embedding_dim)-> (vocab_size,embedding_dim)
        #  (vocab_size,1) * (vocab_size,embedding_dim) -> (vocab_size,embedding_dim) ->  (1,embedding_dim)
        var = torch.sum(weights * torch.pow(emb - mean, 2.), 0, keepdim=True)
        # 求 std
        stddev = torch.sqrt(1e-6 + var)
        # 求 normalized
        normalized_emb = (emb - mean) / stddev
        # torch.FloatTensor -> parameter
        return Parameter(normalized_emb)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False,
                        normalize=True,vocab_freq:Dict[Text,int] = None, keep_prob=1
                        ):
        """

        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            normalize=normalize,
            vocab_freq=vocab_freq,
            keep_prob=keep_prob

        )
        embedding.weight.requires_grad = not freeze
        return embedding


