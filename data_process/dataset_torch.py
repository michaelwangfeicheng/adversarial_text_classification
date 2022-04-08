#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/5 16:08 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/5 16:08   wangfc      1.0         None

Pytorch的数据读取主要包含三个类:
    Dataset
    DataLoader
    DataLoaderIter
这三者大致是一个依次封装的关系: 1.被装进 2., 2.被装进3.





torch.utils.data.Dataset
    是一个抽象类, 自定义的Dataset需要继承它并且实现两个成员方法:
    __getitem__()
    __len__()


torch.utils.data.DataLoader
    主要参数有这么几个:

    dataset : 即上面自定义的dataset.
    collate_fn: 这个函数用来打包batch, 后面详细讲.
    num_worker: 非常简单的多线程方法, 只要设置为>=1, 就可以多线程预读数据啦.

    1.定义了一堆成员变量, 到时候赋给DataLoaderIter,
    2.然后有一个__iter__() 函数, 把自己 "装进" DataLoaderIter 里面.

    def __iter__(self):
            return DataLoaderIter(self)


torch.utils.data.dataloader.DataLoaderIter




class CustomDataset(Dataset):
   # 自定义自己的dataset

dataset = CustomDataset()
dataloader = Dataloader(dataset, ...)

for data in dataloader:
    # training...
    在for 循环里, 总共有三点操作:
    1. 调用了 dataloader 的__iter__() 方法, 产生了一个DataLoaderIter
    2. 反复调用DataLoaderIter 的__next__()来得到 batch, 具体操作就是, 多次调用dataset的__getitem__()方法 (如果num_worker>0就多线程调用), 然后用collate_fn来把它们打包成batch. 中间还会涉及到shuffle , 以及sample 的方法等, 这里就不多说了.
    3. 当数据读完后, __next__()抛出一个StopIteration异常, for循环结束, dataloader 失效.
"""
from typing import List, Tuple, Any

from tqdm import tqdm
from data_process.vocab_utils import UNK, PAD
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
import numpy as np


import logging
logger = logging.getLogger(__name__)



class TextClassifierDataset(Dataset):
    def __init__(self,path,tokenizer,vocab,unk=UNK, pad = PAD, pad_size=32,*args,**kwargs):
        super(TextClassifierDataset, self).__init__(*args,**kwargs)
        self.path = path
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.unk =unk
        self.pad =pad
        self.pad_size =pad_size
        self.data = self._load_data(self.path,self.tokenizer,self.vocab,self.unk,self.pad,self.pad_size)


    def __getitem__(self, item):
        token_ids = self.data[0][item]
        label = self.data[1][item]
        seq_len = self.data[2][item]
        return (token_ids, label, seq_len) # {'token_ids':token_ids,"label":label,'seq_len':seq_len}

    def __len__(self):
        return self.data[0].__len__()


    @staticmethod
    def _load_data(path, tokenizer,vocab ,unk=UNK, pad = PAD, pad_size=32)-> List[Any]:
        labels_ls = []
        token_ids_ls = []
        seq_len_ls = []
        with open(path, 'r', encoding='UTF-8') as f:
            line_num=0
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([pad] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(unk)))

                token_ids_ls.append(words_line)
                labels_ls.append(int(label))
                seq_len_ls.append(seq_len)
                # contents.append((words_line, int(label), seq_len))
                if line_num <5:
                    logger.info(f"content={content},label={label},seq_len={seq_len},token_ids={words_line}")
                line_num+=1
                # if line_num>20000:
                #     break

        # 转换为 numpy
        token_ids_na = np.array(token_ids_ls,dtype=np.int64)
        labels_na = np.array(labels_ls,dtype=np.int64)
        seq_len_na = np.array(seq_len_ls,dtype=np.int64)
        data =  (token_ids_na,labels_na ,seq_len_na)
        logger.info(f"加载数据共{token_ids_na.__len__()} from {path}")
        return data # [([...], 0), ([...], 1), ...]



class DataProvider:
    def __init__(self, batch_size, is_cuda):
        self.batch_size = batch_size
        self.dataset = Dataset_triple(self.batch_size,
                                      transform_=transforms.Compose(
                                     [transforms.Scale([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])]),
                                      )
        self.is_cuda = is_cuda  # 是否将batch放到gpu上
        self.dataiter = None
        self.iteration = 0  # 当前epoch的batch数
        self.epoch = 0  # 统计训练了多少个epoch

    def build(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.dataiter = DataLoaderIter(dataloader)

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iteration += 1

            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch

        except StopIteration:  # 一个epoch结束后reload
            self.epoch += 1
            self.build()
            self.iteration = 1  # reset and return the 1st batch

            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch
