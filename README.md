# adversarial_text_classification
文本分类的对抗训练算法

## Requirements
* python >= 3.7 
* Pytorch >= v1.7.1
* tqdm  
* sklearn  
* tensorboardX


## End-to-end THUCNews Adversarial training Classification

### data
The directory `/corpus/THUCNews` contains the raw THUCNews data.
from [5],从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分:

| 数据集| 数据量 |  
| --- | ---|  
|训练集 | 18万|  
|验证集 | 1万|  
|测试集 | 1万|  

### Train classifier

通过对文本经过embedding 后的输出计算其扰动，生成对抗训练样本，计算对抗训练的adv_loss,将其作为一种 regularization,
加入到原来模型的 loss 中进行训练

```bash
train fgsm
$ python run_text_classifier_task.py \
    --mode=train \
    --cuda-no=0  \
    --adversarial-train-mode=fgsm \

test fgsm
$ python run_text_classifier_task.py \
    --mode=test \
    --cuda-no=0  \
    --adversarial-train-mode=fgsm \


train pgd
$ python run_text_classifier_task.py \
    --mode=train \
    --cuda-no=0  \
    --adversarial-train-mode=pgd \

test fgsm
$ python run_text_classifier_task.py \
    --mode=test \
    --cuda-no=0  \
    --adversarial-train-mode=pgd \


train pgd
$ python run_text_classifier_task.py \
    --mode=train \
    --cuda-no=0  \
    --adversarial-train-mode=free \

test fgsm
$ python run_text_classifier_task.py \
    --mode=test \
    --cuda-no=0  \
    --adversarial-train-mode=free\

```

## evaluation
模型     |precision|recall	|F1 
---     |---      |---    |---
baseline|	0.9118|0.9112 |	0.9110
PGD	    |0.9039   |0.9029  |0.9030
Free	|0.8988   |0.8967  |0.8969
FGSM	|0.9035	  |0.9026  |0.9026



## Reference
[1] Explaining and Harnessing Adversarial Examples.
arxiv.org/abs/1412.6572
[2] Adversarial Training Methods for Semi-Supervised Text Classification.
arxiv.org/abs/1605.07725
[3] Fast is better than free: Revisiting adversarial training
arxiv.org/abs/2001.03994
[4] https://github.com/locuslab/fast_adversarial
[5] https://github.com/649453932/Chinese-Text-Classification-Pytorch

## Contact for Issues
wangfeicheng, wang_feicheng@163.com

