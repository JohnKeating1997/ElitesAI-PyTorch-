# 文本预处理
## 关于re正则
推荐学习 <https://blog.csdn.net/qq_41185868/article/details/96422320#3%E3%80%81%E6%A3%80%E7%B4%A2%E5%92%8C%E6%9B%BF%E6%8D%A2>

>```
>lines = [re.sub(﻿'[^a-z]+'﻿, ' '﻿, line.strip(﻿)﻿.lower(﻿)﻿) for line in f]﻿

上面这行代码的正则部分为
>```
>re.sub(﻿'[^a-z]+'﻿, ' '﻿, str﻿)﻿

re.sub()函数是用来字符串替换的函数﻿

'[^a-z]+' 注意这里的^是非的意思，就是说非a-z字符串﻿

上面句子的含义是：将字符串str中的非小写字母开头的字符串以空格代替

## 字典
```
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False): 
    """
    min_freq 最小词频，小于就不加入字典
    """
        counter = count_corpus(tokens)  # : 
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token): #很好的列表转成反字典的语法
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens): # 填充字典
        if not isinstance(tokens, (list, tuple)): #如果传入的tokens不是个列表或元组（也就是单个字词）
            return self.token_to_idx.get(tokens, self.unk)

        #如果是列表或元组，则对每一个元素进行此方法
        return [self.__getitem__(token) for token in tokens]

        #这个return用得非常好，因为它可以递归，如果tokens有两层及以上的结构它也能把最小单元(token)拿出来

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数

```

测试一下Vocab()
```
vocab = Vocab(tokens)   #tokens在前面定义过（略）
print(list(vocab.token_to_idx.items())[0:10])
```
[('', 0), ('the', 1), ('time', 2), ('machine', 3), ('by', 4), ('h', 5), ('g', 6), ('wells', 7), ('i', 8), ('traveller', 9)]

## Tips：

1. enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

2. dict.get(key, default=None)

   key -- 字典中要查找的键。

   default -- 如果指定键的值不存在时，返回该默认值。
3. isinstance(object, classinfo)  类似type

   如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False.
4. dict.items()

   返回可遍历的(键, 值) 元组数组。

5. 展开一个n维列表

   把sentences里面所有的word展开成一个一维列表tokens（pytorch里可以用tensor.view(-1)来实现）

   sentences 是一个2维列表
   ```
   sentences = [[sentence1],[sentense2],...]

   sentence1 = [word1,word2,...]
   ```
   一行代码实现：
   
   >```
   >tokens = [tk for st in sentences for tk in st]
6. collections.Counter()

   快速的统计一个一维列表里面每个元素出现的次数，并返回一个字典{元素：出现次数}


## 用现有工具进行分词

自己实现的分词方式非常简单，它至少有以下几个缺点:

1. 标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
2. 类似“shouldn't", "doesn't"这样的词会被错误地处理
3. 类似"Mr.", "Dr."这样的词会被错误地处理

事实上，有一些现有的工具可以很好地进行分词，我们在这里简单介绍其中的两个：spaCy和NLTK。

我不做nlp 先略过了

