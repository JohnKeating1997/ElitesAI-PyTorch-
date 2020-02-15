# 语言模型
一段自然语言文本可以看作是一个离散时间序列，给定一个长度为$T$的词的序列$w_1,w_2,\ldots,w_T$，语言模型的目标就是评估该序列是否合理，即计算该序列的概率：

$$P(w_1, w_2, \ldots, w_T).$$

这里介绍基于**统计**的语言模型，主要是n元语法（n-gram）

假设序列中的每个词$w_1,w_2,\ldots,w_T$是**依次生成**的，我们有
$$\begin{align*}
P(w_1, w_2, \ldots, w_T)
&= \prod_{t=1}^T P(w_t \mid w_1, \ldots, w_{t-1})\\
&= P(w_1)P(w_2 \mid w_1) \cdots P(w_T \mid w_1w_2\cdots w_{T-1})
\end{align*}$$
