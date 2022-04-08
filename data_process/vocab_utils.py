#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/4/5 10:43 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/5 10:43   wangfc      1.0         None
"""

import os
from typing import Dict, Text
import pickle as pkl
from tqdm import tqdm



MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(vocab_path, file_path=None, tokenizer=None, max_size=MAX_VOCAB_SIZE, min_freq=1)-> [Dict[Text,int],Dict[Text,int]]:
    """
    vocab :  word2index
    vocab_freq:  word2freq
    """
    from utils.io import get_file_dir
    vocab_freq_path = get_file_dir(file=vocab_path,filename="vocab_freq.pkl")
    if os.path.exists(vocab_path) and os.path.exists(vocab_freq_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
        vocab_freq = pkl.load(open(vocab_freq_path, 'rb'))
    else:
        vocab,vocab_freq = _build_vocab(file_path, tokenizer=tokenizer, max_size=max_size, min_freq=min_freq)
        pkl.dump(vocab, open(vocab_path, 'wb'))
        pkl.dump(vocab_freq,open(vocab_freq_path, 'wb'))
    assert vocab.__len__() == vocab_freq.__len__()
    assert vocab.keys()== vocab_freq.keys()
    print(f"Vocab size: {len(vocab)}")
    return vocab ,vocab_freq


def _build_vocab(file_path, tokenizer, max_size, min_freq)-> [Dict[Text,int],Dict[Text,int]]:
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        vocab_list.extend([(UNK,1),(PAD,1)])
        vocab_freq_dict = {k:v for k,v in vocab_list}

    return vocab_dic, vocab_freq_dict
