# coding: UTF-8
from typing import Dict, Text

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

from models.models_torch.model_configs import ModelConfig


class Config(ModelConfig):

    """配置参数"""
    def __init__(self, dataset, embedding,model_name = 'TextCNN',adversarial_train_mode = None ):
        self.model_name = model_name
        self.adversarial_train_mode = adversarial_train_mode
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)







'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # 初始化卷积：利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的n-gram），从而能够更好地捕捉局部相关性
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        # conv(x) : 对文本序列做 卷积
        # (batch_size,1,sequence_size,embedding_size) -> (batch_size,num_filters,sequence_size,1) -> (batch_size,num_filters,sequence_size)
        x = F.relu(conv(x)).squeeze(3)
        # 沿着 sequence 方向进行 max_pooling：  (batch_size,num_filters,1) -> (batch_size,num_filters)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """
        input x : token_ids,seq_length
        """
        # embeddings  = (batch_size,sequence_size,embedding_size)
        if isinstance(x,tuple) or isinstance(x,list):
            inputs = x[0]
        elif isinstance(x,dict):
            inputs = x['token_ids']
        out = self.embedding(inputs)
        # unsqueezed  ->  (batch_size,1,sequence_size,embedding_size)
        out = out.unsqueeze(1)
        # 先做卷积和 max_pooling，再进行对这多个不同size的conv结果进行最后维度的拼接
        # (batch_size,1,nums_filter * len(filter_sizes))
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class TextCNNAT(Model):
    """
    支持 adversarial training
    """
    def __init__(self,config,vocab_freq=None):
        super(Model, self).__init__()
        self.vocab_freq = vocab_freq
        self.embedding =  self._build_embedding(config=config,vocab_freq = self.vocab_freq)

        self.embedding_dim= self.embedding.embedding_dim

        # 初始化卷积：利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的n-gram），从而能够更好地捕捉局部相关性
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)



    def _build_embedding(self,config:ModelConfig,vocab_freq:Dict[Text,int]):
        if config.adversarial_train_mode is None:
            if config.embedding_pretrained is not None:
                embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            else:
                embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        else:
            # 使用 normalized embedding
            from utils.torch.layers import NormalizedEmbedding
            if config.embedding_pretrained is not None:
                embedding = NormalizedEmbedding.from_pretrained(embeddings=config.embedding_pretrained, freeze=False,
                                                                padding_idx=config.n_vocab - 1,vocab_freq=vocab_freq
                                                                )
            else:
                embedding = NormalizedEmbedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        return embedding

    def forward(self, x,input_embedding=False):
        """
        input x : token_ids,seq_length
        """
        if not input_embedding:
            # embeddings  = (batch_size,sequence_size,embedding_size)
            if isinstance(x,tuple) or isinstance(x,list):
                inputs = x[0]
            elif isinstance(x,dict):
                inputs = x['token_ids']
            elif isinstance(x,torch.Tensor):
                inputs = x
            out = self.embedding(inputs)
        else:
            out = x
        # unsqueezed  ->  (batch_size,1,sequence_size,embedding_size)
        out = out.unsqueeze(1)
        # 先做卷积和 max_pooling，再进行对这多个不同size的conv结果进行最后维度的拼接
        # (batch_size,1,nums_filter * len(filter_sizes))
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out





class TextCNNV1(torch.nn.Module):
    def __init__(self, output_size, vocab_size, embedding_dim, padding_idx, n_filters, filter_sizes, dropout):
        super(TextCNNV1, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                            padding_idx=padding_idx)
        self.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=n_filters,
                                     kernel_size=(filter_sizes[0], embedding_dim))
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n_filters,
                                     kernel_size=(filter_sizes[1], embedding_dim))
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=n_filters,
                                     kernel_size=(filter_sizes[2], embedding_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_features=n_filters * 3, out_features=output_size)

    def forward(self, batch):
        # 输入的 text = [batch_size, ]
        # 输入 text = [max_sequence,batch_size] -> [max_sequence,batch_size, embedding_size]
        embedded = self.embedding(batch.text)
        #  [max_sequence,batch_size, embedding_size] ->  [batch_size,max_sequence, embedding_size]
        embedded = embedded.permute((1, 0, 2))
        # [batch_size,max_sequence, embedding_size] -> [batch_size,1, max_sequence, embedding_size]
        embedded.unsqueeze_(dim=1)
        # 经过 conv_0 层:
        # input =  [batch_size,1, max_sequence, embedding_size] = [batch_size,in_channels,H,W]
        # output = [batch_size,num_filters, max_sequence, 1] =  [batch_size,output_channels,H-filter_size +1,1]

        conv0_out = F.relu(self.conv0(embedded).squeeze(dim=-1))
        conv1_out = F.relu(self.conv1(embedded).squeeze(dim=-1))
        conv2_out = F.relu(self.conv2(embedded).squeeze(dim=-1))

        # avg_pool1d： kernel_size = H 窗口的尺度
        # input： [batch_size, in_channels,H]
        # output: [batch_size,in_channels,1]
        # [batch_size,num_filters, max_sequence] -> [batch_size,num_filters, 1]
        pool0_out = F.avg_pool1d(conv0_out, kernel_size=conv0_out.shape[-1]).squeeze(dim=-1)
        pool1_out = F.avg_pool1d(conv0_out, kernel_size=conv1_out.shape[-1]).squeeze(dim=-1)
        pool2_out = F.avg_pool1d(conv0_out, kernel_size=conv2_out.shape[-1]).squeeze(dim=-1)

        # [batch_size,num_filters*3]
        cat = torch.cat([pool0_out, pool1_out, pool2_out], dim=1)
        # 为防止过拟合
        dropouted = self.dropout(cat)
        output = self.linear(dropouted)
        return output


class TextCNNV2(torch.nn.Module):
    def __init__(self, output_size, vocab_size, embedding_dim, padding_idx, n_filters, filter_sizes, dropout):
        super(TextCNNV1, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                            padding_idx=padding_idx)

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, embedding_dim))
             for filter_size in filter_sizes])
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_features=n_filters * len(filter_sizes), out_features=output_size)

    def forward(self, batch):
        # 输入的 text = [batch_size, ]
        # 输入 text = [max_sequence,batch_size] -> [max_sequence,batch_size, embedding_size]
        embedded = self.embedding(batch.text)
        #  [max_sequence,batch_size, embedding_size] ->  [batch_size,max_sequence, embedding_size]
        embedded = embedded.permute((1, 0, 2))
        # [batch_size,max_sequence, embedding_size] -> [batch_size,1, max_sequence, embedding_size]
        embedded.unsqueeze_(dim=1)
        # 经过 conv_0 层: [batch_size,1, max_sequence, embedding_size] -> [batch_size,num_filters, max_sequence, 1]
        conv_outs = [F.relu(conv(embedded).squeeze(dim=-1)) for conv in self.convs]

        # [batch_size,num_filters, max_sequence] -> [batch_size,num_filters, 1]
        pool_outputs = [F.avg_pool1d(conv_out, kernel_size=conv_out.shape[-1]).squeeze(dim=-1) for conv_out in
                        conv_outs]

        # [batch_size,num_filters*3]
        cat = torch.cat(pool_outputs, dim=1)
        # 为防止过拟合
        dropouted = self.dropout(cat)
        output = self.linear(dropouted)
        return output



class TextCNN1d(torch.nn.Module):
    def __init__(self, output_size, vocab_size, embedding_dim, padding_idx, n_filters, filter_sizes, dropout):
        super(TextCNN1d, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                            padding_idx=padding_idx)
        # 使用 Conv1d 进行卷积操作：
        # input = [Batch_size,in_channels,H]
        # output = [Batch_size,n_filter, H-filter_size+1]
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_size)
             for filter_size in filter_sizes])
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_features=n_filters * len(filter_sizes), out_features=output_size)

    def forward(self, batch):
        # 输入的 text = [batch_size, ]
        # 输入 text = [max_sequence,batch_size] -> [max_sequence,batch_size, embedding_size]
        embedded = self.embedding(batch.text)
        #  [max_sequence,batch_size, embedding_size] ->  [batch_size,embedding_size,max_sequence]
        embedded = embedded.permute((1, 2, 0))

        # 经过 conv_0 层: [batch_size,embedding_size,max_sequence] -> [batch_size,num_filters, max_sequence]
        conv_outs = [F.relu(conv(embedded).squeeze(dim=-1)) for conv in self.convs]

        # [batch_size,num_filters, max_sequence] -> [batch_size,num_filters, 1]
        pool_outputs = [F.avg_pool1d(conv_out, kernel_size=conv_out.shape[-1]).squeeze(dim=-1) for conv_out in
                        conv_outs]

        # [batch_size,num_filters*3]
        cat = torch.cat(pool_outputs, dim=1)
        # 为防止过拟合
        dropouted = self.dropout(cat)
        output = self.linear(dropouted)
        return output
