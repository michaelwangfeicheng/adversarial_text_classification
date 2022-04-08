#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/3/2 10:42 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/2 10:42   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from enum import Enum
from utils.utils import pytorch_cos_sim


class DistanceMetrics(Enum):
    """"""
    # 沿着某个抽计算 cosine 相似度：consine 距离： [0,2] ->[0,1]
    COSINE = lambda x,y: (1- F.cosine_similarity(x,y,dim=1))/2.0
    EUCLIDEAN = lambda x,y: F.pairwise_distance(x,y,p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)



class TripletLoss(nn.Module):
    # 自定义一种loss
    def __init__(self,distance_metric=DistanceMetrics.COSINE,margin=0.5):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self,triplet_embedding:Tensor=None):
        # 可以设计为 anchor = (batch_size,embedding_size)
        # anchor_embedding, pos_embedding, neg_embedding = triplet_embedding
        # 也可以设计为 triplet = (batch_size, 3, embedding_size)
        assert triplet_embedding.shape[1] ==3
        # 提取对应的 embedding
        anchor_embedding = triplet_embedding[:, 0, :]
        pos_embedding = triplet_embedding[:, 1, :]
        neg_embedding = triplet_embedding[:, 2, :]

        anchor2pos_distances = self.distance_metric(anchor_embedding,pos_embedding)
        anchor2neg_distances = self.distance_metric(anchor_embedding,neg_embedding)
        losses = F.relu(anchor2pos_distances-anchor2neg_distances+self.margin)
        losses_mean = losses.mean()
        return losses_mean



class BatchHardTripletLoss(nn.Module):
    """
    @author:wangfc
    @desc:
    这种方法属于 online triplet mining
        1) 形成 （batch_size * batch_size* batch_size）的 任意三元组
        2）使用每个句子对应的label，筛选 有效的三元组： anchor，positive,negative
        3) 计算距离，找到 其中 hard triplet: |e_a - e_p| > |e_a -e_n|

    @version：
    @time:2021/3/5 16:18

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self,distance_metric= 'cosine',margin=0.5):
        super(BatchHardTripletLoss, self).__init__()
        if distance_metric == 'cosine':
            self.distance_metric = BatchHardTripletLossDistanceFunction.cosine_distance
        elif distance_metric == 'eucledian':
            self.distance_metric = BatchHardTripletLossDistanceFunction.eucledian_distance
        self.margin = margin

    def forward(self,embeddings:Tensor,labels:Tensor):
        loss = self.batch_hard_triplet_loss(embeddings=embeddings,labels=labels)
        return loss

    # Hard Triplet Loss
    # Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    # Blog post: https://omoindrot.github.io/triplet-loss
    def batch_hard_triplet_loss(self, embeddings: Tensor,labels: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """

        # Get the pairwise distance matrix :
        # embeddings = (batch_size,embedding_size)  -> (batch_size,batch_size )
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        #  mask_anchor_positive = (batch_size,batch_size )
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        # anchor_positive_dist =  (batch_size,batch_size )
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # 取 anchor_positive_dist 最大值：
        # hardest_positive_dist =  (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        # mask_anchor_negative = (batch_size,batch_size )
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        # 对 非 mask_anchor_negative 按行 增加一个 行最大值，再按行取最小值，就得到 anchor_negative_dist 的最小值
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # hardest_negative_dist =  (batch_size,1)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        distances = hardest_positive_dist - hardest_negative_dist + self.margin
        # tl[tl < 0] = 0
        # triplet_loss = tl.mean()
        losses = F.relu(input=distances)
        triplet_loss = losses.mean()
        return triplet_loss


    @staticmethod
    def get_anchor_positive_triplet_mask(labels):
        """
        计算每个 anchor 对应的 positive_mask
        Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        # 新建单位矩阵
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        # 去除对角线的部分
        return labels_equal & indices_not_equal


    @staticmethod
    def get_anchor_negative_triplet_mask(labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return ~ labels_equal



class BatchHardTripletLossDistanceFunction:
    """
    This class defines distance functions, that can be used with Batch[All/Hard/SemiHard]TripletLoss
    """
    @staticmethod
    def cosine_distance(embeddings):
        """
        @author:wangfc27441
        @desc:
        因为该 cosine_distance 是用在 loss函数中的，衡量预测和实际的embedding的 差异性

        一般的 cosine_distance 可以定义为：
        (1+cosine_similarity) * 0.5，使得新的 cosine_distance 在[0,1] 之间

        @version：
        @time:2020/10/27 9:58

        """
        """
        Compute the 2D matrix of cosine distances (1-cosine_similarity) between all embeddings.
        """
        return  0.5*(1 - pytorch_cos_sim(embeddings, embeddings))


    @staticmethod
    def eucledian_distance(embeddings, squared=False):
        """
        Compute the 2D matrix of eucledian distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # (batch_size,batch_size)
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances







