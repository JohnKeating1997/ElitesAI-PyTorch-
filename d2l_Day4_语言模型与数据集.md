# 语言模型（这里的模型与离散时间序列类似）
一段自然语言文本可以看作是一个**离散时间序列**，给定一个长度为$T$的词的序列$w_1,w_2,\ldots,w_T$，语言模型的目标就是评估该序列是否合理，即计算该序列的概率：

$$P(w_1, w_2, \ldots, w_T).$$

这里介绍基于**统计**的语言模型，主要是n元语法（n-gram）

假设序列中的每个词$w_1,w_2,\ldots,w_T$是**依次生成**的，我们有
$$P(w_1, w_2, \ldots, w_T)
&= \prod_{t=1}^T P(w_t \mid w_1, \ldots, w_{t-1})\\$$

$$&= P(w_1)P(w_2 \mid w_1) \cdots P(w_T \mid w_1w_2\cdots w_{T-1})$$


例如，一段含有4个词的文本序列的概率
$$P(w_1, w_2, w_3, w_4) =  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3).$$
语言模型的参数就是词的**概率**(频率)以及给定前几个词情况下的**条件概率**。设训练数据集为一个大型文本语料库，如维基百科的所有条目，词的概率可以通过该词在训练数据集中的相对**词频**来计算，例如，$w_1$的概率可以计算为：
$$\hat P(w_1) = \frac{n(w_1)}{n}$$
其中：

   $n(w_1)$是$w_1$作为第一个词的文本的数量，$n$为语料库中文本的总数量。

类似地，给定$w_1$情况下，$w_2$的条件概率可以计算为：
$$\hat P(w_2 \mid w_1) = \frac{n(w_1, w_2)}{n(w_1)}$$
其中$n(w_1,w_2)$是语料库中以$w_1$为第一个词，$w_2$为第二个词的文本数量

## n元语法（n-grams)
序列长度增加，计算和存储多个词共同出现的概率的复杂度会呈指数级增加。例如，一个语料库中有n个词,那么计算$\hat P(w_1)$,$\hat P(w_2 \mid w_1)$，$\hat P(w_3 \mid w_1,w_2)$ 所要计算的概率数量（即参数空间）分别为$n$,$n^2$,$n^3$（即考虑所有的组合数），如果是3元语法，那么参数空间 = $n+n^2+n^3$

n元语法空间从一定程度上 **alleviate** 这一问题，它基于n-1阶马尔科夫链（Markov chain of order n-1），即一个词只与前面n-1的词有关，则一段严格按照$w_1,w_2, \ldots ,w_T$词序的文本出现的概率为
$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T P(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1}) .$$
当较小时，n元语法往往并不准确。例如，在一元语法中，由三个词组成的句子“我爱你”和“你爱我”的概率是一样的（想peach）。然而，当较大时，n元语法需要计算并存储大量的词频和多词相邻频率(computation and storage burden)。

### n元语法存在的问题
1. 参数空间过大（特别是当语料库中的tokens非常多的时候）
2. 数据稀疏（绝大部分排列很少用到甚至不用，比如“在天朝可以访问全世界”、“下岗了我好快乐”，“风马牛”）

# 时序数据的采样
以周杰伦《可爱女人》为例，“想要有直升机 想要和你飞到宇宙去 想要和...”（高中的回忆呀...）

每次随机读取小批量样本和标签。标签序列为这些字符分别在训练集中的下一个字符。例如，如果时间步数为5，有以下可能的样本和标签：

X：“想要有直升”，Y：“要有直升机”

X：“要有直升机”，Y：“有直升机，”

......

X：“你飞到宇宙”，Y：“飞到宇宙去”

如果序列的长度为$T$，时间步数为$n$，那么一共有$T-n$个合法的样本，但是这些样本有大量的重合，我们通常采用更加高效的采样方式。
## 随机采样
代码实现：其中，batch_size是每个batch包含的sample数，num_steps是每个sample所包含的时间步数(length)。

在随机采样中，每个批量的每个样本是原始序列上任意截取的一段序列，**相邻的两个随机小批量的样本**  以及  **一个小批量内的各个样本**在原始序列上的位置不一定相毗邻。
```
import torch
import random
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为要给标签留下最后一个词
    num_examples = (len(corpus_indices) - 1) // num_steps  # //整除，即下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标，把样本切碎
    random.shuffle(example_indices)  把切碎的样本打乱

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)  

```
注意这里的yield实际上已经把date_iter_random这个**function**变成一个**可迭代对象**了

这里用**yield**不用**return**的原因是，用了return这个函数就终止了(for只跑了一次),而用了yield只是**中止一次循环** *（注意是下面 for X,Y in data_iter_random(...)的那个for循环，不是def里的循环）*

测试一下这个函数，我们输入从0到29的连续整数作为一个人工序列，batch_siza=2，num_steps=6，打印随机采样每次读取的小批量样本的输入X和标签Y。
```
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6): #带yield的函数
    print('X: ', X, '\nY:', Y, '\n')
```
结果如下
```
X:  tensor([[ 6,  7,  8,  9, 10, 11],       #可以看到同一batch里的样本并不相连
        [12, 13, 14, 15, 16, 17]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],       #但是Y（标签）跟样本（X）相连的
        [13, 14, 15, 16, 17, 18]]) 

X:  tensor([[ 0,  1,  2,  3,  4,  5],      #可以看到第二个batch的X和第一个batch的样本也不相连
        [18, 19, 20, 21, 22, 23]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [19, 20, 21, 22, 23, 24]])     
```
## 相邻采样
在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。
```
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
   #比如我每个batch要2个样本，就把整个样本剁成两段，这样每一batch都在这两端里分别取一个，相邻的两个batch的sample是连续的
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )  
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```
同样的设置下，打印相邻采样每次读取的小批量样本的输入X和标签Y。相邻的两个随机小批量在原始序列上的位置相毗邻。
```
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```
输出结果如下：
```
X:  tensor([[ 0,  1,  2,  3,  4,  5],
        [15, 16, 17, 18, 19, 20]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [16, 17, 18, 19, 20, 21]]) 

X:  tensor([[ 6,  7,  8,  9, 10, 11],
        [21, 22, 23, 24, 25, 26]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],
        [22, 23, 24, 25, 26, 27]]) 
```
## Tips
1. class set([iterable])
   创建一个无序去重元素集，可进行关系测试，删除重复数据，*还可以计算交集、差集、并集等*。
2. yield把funciton变成一个迭代器，省去了另创建列表“存储”每次产生的值所带来的内存负担，也增强了代码的可读性。
