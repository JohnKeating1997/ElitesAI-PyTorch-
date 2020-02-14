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

批量梯度下降法是最原始的形式，它是指在每一次迭代时使用所有样本来进行梯度的更新。
**优点：**
（1）一次迭代是对所有样本进行计算，此时利用矩阵进行操作，实现了并行。
（2）由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。当目标函数为凸函数时，BGD一定能够得到全局最优。
 **缺点：**
 （1）当样本数目 m 很大时，每迭代一步都需要对所有样本计算，训练过程会很慢。
  从迭代的次数上来看，BGD迭代的次数相对较少。其迭代的收敛曲线示意图可以表示如下：
  ### 2、随机梯度下降（Stochastic Gradient Descent，SGD）
  随机梯度下降法不同于批量梯度下降，随机梯度下降是每次迭代使用一个样本来对参数进行更新。使得训练速度加快。
  **优点：**
  （1）由于不是在全部训练数据上的损失函数，而是在每轮迭代中，随机优化某一条训练数据上的损失函数，这样每一轮参数的更新速度大大加快。
  **缺点：**
  （1）准确度下降。由于即使在目标函数为强凸函数的情况下，SGD仍旧无法做到线性收敛。
  （2）可能会收敛到局部最优，由于单个样本并不能代表全体样本的趋势。
  （3）不易于并行实现。
  **解释一下为什么SGD收敛速度比BGD要快：**
  答：这里我们假设有30W个样本，对于BGD而言，每次迭代需要计算30W个样本才能对参数进行一次更新，需要求得最小值可能需要多次迭代（假设这里是10）；而对于SGD，每次更新参数只需要一个样本，因此若使用这30W个样本进行参数更新，则参数会被更新（迭代）30W次，而这期间，SGD就能保证能够收敛到一个合适的最小值上了。也就是说，在收敛时，BGD计算了 10×30W 次，而SGD只计算了 1×30W 次。
  ### 3、小批量梯度下降（Mini-Batch Gradient Descent, MBGD）
  小批量梯度下降，是对批量梯度下降以及随机梯度下降的一个折中办法。其思想是：每次迭代 使用 ** batch_size** 个样本来对参数进行更新。
  **优点：**
  （1）通过矩阵运算，每次在一个batch上优化神经网络参数并不会比单个数据慢太多。
  （2）每次使用一个batch可以大大减小收敛所需要的迭代次数，同时可以使收敛到的结果更加接近梯度下降的效果。(比如上例中的30W，设置batch_size=100时，需要迭代3000次，远小于SGD的30W次)
  （3）可实现并行化。
  **缺点：**
  （1）batch_size的不当选择可能会带来一些问题。
  **batcha_size的选择带来的影响：**
  （1）在合理地范围内，增大batch_size的好处：
    a. 内存利用率提高了，大矩阵乘法的并行化效率提高。
    b. 跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
    c. 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。
  （2）盲目增大batch_size的坏处：
    a. 内存利用率提高了，但是内存容量可能撑不住了。
    b. 跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
    c. Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。

