#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/28 14:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/28 14:51   wangfc      1.0         None
"""
import os
import sys




def run_text_classifier_task_torch():

    try:
        from cli.arguments.task_argument_parser import create_text_classifier_task_argument_parser
        from tasks.torch_task import TextClassifierTaskTorch
        from models.models_torch.classifiers.TextCNN import Config as TextCNNConfig
        from models.models_torch.classifiers.TextCNN import Model as TextCNNModel
        from tokenizations.tokenizatoions_utils import basic_tokenizer

        paser = create_text_classifier_task_argument_parser(mode='train',adversarial_train_mode='free',cuda_no=0)
        args = paser.parse_args()
        model_name = args.model
        adversarial_train_mode = args.adversarial_train_mode
        optimizer_name = args.optimizer_name
        output_dir = args.output_dir
        random_sampling = args.random_sampling
        test_no = "02"

        if adversarial_train_mode is None:
            output_dir = os.path.join(output_dir, model_name)
        else:
            output_dir = os.path.join(output_dir, f"{model_name}_{adversarial_train_mode}")

        from utils.common import init_logger
        log_dir = os.path.join(output_dir, 'log')
        logger = init_logger(output_dir=log_dir, log_filename=args.mode, log_level='info')

        dataset = 'corpus/THUCNews'  # 数据集
        # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
        embedding = 'embedding_SougouNews.npz'

        if args.model in ["TextCNN", "TextCNNAT"]:
            model_config = TextCNNConfig(dataset=dataset, embedding=embedding,
                                         model_name=model_name,
                                         adversarial_train_mode=adversarial_train_mode)

        data_dir = os.path.join(dataset, 'data')
        num_train_epochs = model_config.num_epochs
        batch_size = model_config.batch_size
        learning_rate = model_config.learning_rate
        tokenizer = basic_tokenizer(ues_word=False)

        text_classifier_task_torch = TextClassifierTaskTorch(mode=args.mode,
                                                             cuda_no=args.cuda_no,
                                                             data_dir=data_dir, output_dir=output_dir,
                                                             model_config=model_config,
                                                             tokenizer=tokenizer,
                                                             vocab_path=model_config.vocab_path,
                                                             optimizer_name=optimizer_name, learning_rate=learning_rate,
                                                             train_filename='train.txt', dev_filename='dev.txt',
                                                             test_filename='test.txt',
                                                             num_train_epochs=num_train_epochs,
                                                             train_batch_size=batch_size,
                                                             adversarial_train_mode=adversarial_train_mode,
                                                             random_sampling=random_sampling,
                                                             attack_iters = args.attack_iters,
                                                             minibatch_replays = args.minibatch_replays
                                                             )
        if args.mode == 'train':
            text_classifier_task_torch.train()
        else:
            text_classifier_task_torch.test()
    except Exception as e:
        logger.error(msg=e,exc_info=True)
        sys.exit(0)


if __name__ == '__main__':
    # main()
    # run_chinese_text_classifier_task()
    run_text_classifier_task_torch()
