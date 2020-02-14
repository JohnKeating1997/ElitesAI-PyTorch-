# 一、线性回归
## pytorch内的一些函数
### torch.Tensor
是一种包含单一数据类型元素的多维矩阵，定义了7种CPU tensor和8种GPU tensor类型。
### torch.tensor和torch.Tensor
在Pytorch中，Tensor和tensor都用于生成新的张量。
torch.Tensor()是一个类，是默认张量类型torch.FloatTensor()的别名，torch.Tensor([1,2]) 会调用Tensor类的构造函数__init__，生成单精度浮点（float32）类型的张量。

> a=torch.Tensor([1,2])
> a.type()
> #'torch.FloatTensor'

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
```
x = tensor.ones(2,4,requires_grad=True)
```
2.如果想改变这个属性，就调用tensor.requires_grad_()方法：
```
x.requires_grad_(False)
```
