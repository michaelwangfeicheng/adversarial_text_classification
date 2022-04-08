#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/1 15:09 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/1 15:09   wangfc      1.0         None
"""
import argparse


def create_argument_parser(mode='test', log_level='info',
                           port=8015,
                           num_processes=1,
                           gpu_memory_config="1:128"
                           ) -> argparse.ArgumentParser:
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        prog="Hsnlp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Hsnlp command line interface.",
    )
    parser.add_argument('-m', '--mode', default=mode, help="设置模式", type=str)
    parser.add_argument('-l', '--log_level', default=log_level, help="设置log_level", type=str)
    parser.add_argument('-g', '--gpu_memory_config', default=gpu_memory_config, help="设置 gpu_memory_config", type=str)
    parser.add_argument('-p', '--port', default=port, help="设置端口", type=str)
    parser.add_argument('-n', '--num_processes', default=num_processes, help="设置 num_processes", type=int)


    # test_result_file_suffix = "result_03"
    # use_intent_attribute_regex_classifier = True
    # hsnlp_word_segment_url = "http://10.20.33.3:8017/hsnlp/faq/sentenceProcess"

    return parser


def create_text_classifier_task_argument_parser(mode='train',adversarial_train_mode='fgsm',cuda_no=0)-> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--mode', type=str, default=mode)
    parser.add_argument('--cuda-no', type=int, default=cuda_no)
    parser.add_argument('-m','--model', type=str, default='TextCNNAT',help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    parser.add_argument('-e','--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('-w','--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('-o','--optimizer-name', default='Adam', type=str, help='The optimizer_name for train')
    parser.add_argument('--random-sampling', default=True, type=bool, help='训练数据是否随机抽样')

    parser.add_argument('--output_dir', default='output/chinese_text_classification', type=str, help='The output dir for train')
    parser.add_argument('-at','--adversarial-train-mode', default=adversarial_train_mode, type=str,help='choose mode of adversarial training: fgsm,free,pgd')
    parser.add_argument('--epsilon', default=8, type=int,help="对抗训练的扰动步长")
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],help='Perturbation initialization method')
    parser.add_argument('--perturb-norm-length', default=5.0,type=float,help='Norm length of adversarial perturbation to be optimized with validation. 5.0 is optimal on IMDB with virtual adversarial training. ')
    parser.add_argument('--adv-reg-coeff', default='1.0', type=float,help='对样训练的产生的loss的 coeff')
    parser.add_argument('--normalize-embeddings', default=True,type=bool,help='是否对 embedding 进行规范化')
    parser.add_argument('--max-grad-norm', default=1.0, type=float, help='Clip the global gradient norm to this value.')
    parser.add_argument('--keep-prob-emb', default=1.0, type=float, help='keep probability on embedding layer. '
                   '0.5 is optimal on IMDB with virtual adversarial training.')
    parser.add_argument('--attack-iters', default=7, type=int, help='pgd Attack iterations')
    parser.add_argument('--minibatch-replays', default=8, type=int, help='free minibatch replays')
    return parser