# 文本预处理/语言模型/循环神经网络
## 关于re正则
推荐学习 <https://blog.csdn.net/qq_41185868/article/details/96422320#3%E3%80%81%E6%A3%80%E7%B4%A2%E5%92%8C%E6%9B%BF%E6%8D%A2>

>```
>lines = [re.sub(﻿'[^a-z]+'﻿, ' '﻿, line.strip(﻿)﻿.lower(﻿)﻿) for line in f]﻿
