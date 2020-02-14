# 一、线性回归
## pytorch内的一些函数
### torch.Tensor
是一种包含单一数据类型元素的多维矩阵，定义了7种CPU tensor和8种GPU tensor类型。
### torch.tensor和torch.Tensor
在Pytorch中，Tensor和tensor都用于生成新的张量。
torch.Tensor()是一个类，是默认张量类型torch.FloatTensor()的别名，torch.Tensor([1,2]) 会调用Tensor类的构造函数__init__，生成单精度浮点（float32）类型的张量。
> ```
> a=torch.Tensor([1,2])
> a.type()   #'torch.FloatTensor'

torch.tensor()是一个函数/方法，函数原型是：
>```
> torch.tensor(data, dtype=None, device=None, requires_grad=False)

其中data可以是：list, tuple, array, scalar等类型。
torch.tensor()可以从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的**torch.LongTensor，torch.FloatTensor，torch.DoubleTensor**。

### 构造初始化tensor
torch.ones()/torch.zeros() 与MATLAB的ones/zeros很接近。初始化生成1或者0
### 均匀分布
torch.rand(*sizes, out=None) → Tensor

返回一个tensor(张量)，包含了从区间[0,1) 的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。

### 标准正态分布 random normal distribution
torch.randn(*sizes, out=None) → Tensor
返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。

### torch.mul(a, b)
是矩阵a和b**对应位**相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的**仍是(1, 2)的矩阵**
### torch.mm(a, b) 正常内积
是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵

### random.shuffle(a)
用于将一个列表中的元素打乱。shuffle() 是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。

### tensor.backward()
backward()是pytorch中提供的函数，配套有require_grad：
1.所有的tensor都有.requires_grad属性,可以设置这个属性.
> ```
> x = tensor.ones(2,4,requires_grad=True)

2.如果想改变这个属性，就调用tensor.requires_grad_()方法：
> ```
> x.requires_grad_(False)

## 批量梯度下降(BGD)、随机梯度下降(SGD)以及小批量梯度下降(MBGD)的理解
[转自](https://www.cnblogs.com/lliuye/p/9451903.html)
梯度下降法作为机器学习中较常使用的优化算法，其有着三种不同的形式：批量梯度下降（Batch Gradient Descent）、随机梯度下降（Stochastic Gradient Descent）以及小批量梯度下降（Mini-Batch Gradient Descent）。其中小批量梯度下降法也常用在**深度学习**中进行模型的训练。
### 1、批量梯度下降（Batch Gradient Descent，BGD）
\frac{\Delta J(\theta_0,\theta_1)}{\Delta \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}
批量梯度下降法是最原始的形式，它是指在每一次迭代时使用所有样本来进行梯度的更新。
  **优点：**
  （1）一次迭代是对所有样本进行计算，此时利用矩阵进行操作，实现了并行。
  （2）由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。当目标函数为凸函数时，BGD一定能够得到全局最优。
 **缺点：**
  （1）当样本数目 m 很大时，每迭代一步都需要对所有样本计算，训练过程会很慢。
  从迭代的次数上来看，BGD迭代的次数相对较少。其迭代的收敛曲线示意图可以表示如下：
  ### 2、随机梯度下降（Stochastic Gradient Descent，SGD）
  随机梯度下降法不同于批量梯度下降，随机梯度下降是每次迭代使用一个样本来对参数进行更新。使得训练速度加快。
