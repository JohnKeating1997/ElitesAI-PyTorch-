# 文本预处理/语言模型/循环神经网络
## 关于re正则
推荐学习 <https://blog.csdn.net/qq_41185868/article/details/96422320#3%E3%80%81%E6%A3%80%E7%B4%A2%E5%92%8C%E6%9B%BF%E6%8D%A2>

>```
>lines = [re.sub(﻿'[^a-z]+'﻿, ' '﻿, line.strip(﻿)﻿.lower(﻿)﻿) for line in f]﻿

# 上面这行代码的正则部分为
>```
>re.sub(﻿'[^a-z]+'﻿, ' '﻿, str﻿)﻿

#re.sub()函数是用来字符串替换的函数﻿

#'[^a-z]+' 注意这里的^是非的意思，就是说非a-z字符串﻿

#上面句子的含义是：将字符串str中的非小写字母开头的字符串以空格代替

## 学到一个很优雅的语法
sentences 是一个2维列表
```
sentences = [[sentence1],[sentense2],...]

sentence1 = [word1,word2]...
```
...
把sentences里面所有的word展开成一个一维列表tokens（pytorch里可以用tensor.view(-1)来实现）
>```
>tokens = [tk for st in sentences for tk in st]

## collections.Counter()
快速的统计一个一维列表里面每个元素出现的次数，并返回一个字典{元素：出现次数}

