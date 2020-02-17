#numpy、torch里的一些数组格式
# np.array
[关于array数据结构方面的机理链接](https://www.runoob.com/numpy/numpy-ndarray-object.html)

numpy 最重要的一个特点是其 N 维数组对象 **ndarray**(n-dimension-array)，它是一系列**同类型数据**的集合，以 **0 下标为开始**进行集合中元素的索引。
**array**是一个**方法**，而**ndarray**是一个**类/对象**
```
import numpy as np
a = np.array([1,2,3,4,5])
print(type(a))   #<class 'numpy.ndarray'>
```
### axis 与 keepdims
numpy 中的axis是由外而内，一层一层 **剥** 开他的心的。这和 切片 ,以及len()函数的逻辑是相通的
```
import numpy as np
a=np.array([[[1],[3]],[[10],[30]],[[100],[300]]])

print(a)
print(np.shape(a))
```
结果显示，最外层axis=0的size是3，最里层axis=2的size是1
```
[[[  1]
  [  3]]

 [[ 10]
  [ 30]]

 [[100]
  [300]]]
(3, 2, 1)
```

#### 栗1：
numpy里带有axis参数的函数，axis = i表示对第i层里的最大单位块做块与块之间的运算,在**keepdims=False**的情况下(**by default**)，**移除第i层```[]```**：
以```sum()```方法为例：
```
a= np.array([1,2,3])   
a.sum(axis = 0)
>>>6
```
因为只有一层壳```[]```，所以直接对这一层里的最大单位快1，2，3做运算；
做完加法后本应是[6]，但是移除最外层壳```[]```后，壳```[]```不存在了，所以返回的是6。

#### 栗2：
```
a= np.array([[1,2],[3,4]]) 
a.sum(axis = 1)
>>>array([3, 7])
```
有两层壳```[]```，第二外层```[]```里的最大单位块有两组（因为有两个第二外层[]），第一组是1，2，第二组是3，4，分别对这两个单位块做块与块之间的运算，第一组结果为1+2=3，第二组结果为3+4=7；
做完加法后本应是```[[3],[7]]```，但是**移除第二外层的壳**```[]```后，原来的两层```[]```变成一层```[]```,所以返回结果为```[3, 7]```。

### size,shape
```size()```：计算数组和矩阵所有数据的个数 
```
a = np.array([[1,2,3],[4,5,6]]) 
np.size(a)   #6 缺省情况下，返回数组和矩阵里所有数据的个数
np.size(a,1)  #3 axis=1的情况下，返回第2层的元素的个数(即[1,2,3]里或[4,5,6]里元素的个数)
```
```shape ()```:得到矩阵每维的大小 
```
print(np.shape(a))
#返回(2,3)
```

### np.array和np.mat
定位不同，ndarray是**n维数组**，相当于扩展版的list；
而matrix只是一个**2维矩阵**，方便做矩阵运算，包含在array里，但与array有**不一样的用法**

1. matrix 和 array 都可以通过objects后面加.T 得到其**转置**。但是 matrix objects 还可以在后面加 .H f得到**共轭矩阵**, 加 .I 得到**逆矩阵**。
2. **点乘和对应位置乘**
mat中 * 就是点乘， np.multiply()是对应位置乘
```
import numpy as np

a=np.mat('4 3; 2 1')
b=np.mat('1 2; 3 4')
print(a)
# [[4 3]
#  [2 1]]
print(b)
# [[1 2]
#  [3 4]]

print(a*b)
# [[13 20]
#  [ 5  8]]

np.multiply(a,b)
# matrix([[ 4,  6],
          [6,  4]])  
```
  而array中， * 代表对应位置乘法，np.dot()代表点乘
```
c=np.array([[4, 3], [2, 1]])
d=np.array([[1, 2], [3, 4]])

print(c*d)
# [[4 6]
#  [6 4]]

print(np.dot(c,d))
# [[13 20]
#  [ 5  8]]
```
#torch中的Tensor(张量)
###几个数学概念：

标量（Scalar）是只有大小，没有方向的量，如1，2，3等

向量（Vector）是有大小和方向的量，其实就是一串数字，如(1,2)

矩阵（Matrix）是好几个向量拍成一排合并而成的一堆数字，如```[1,2;3,4]```

**与张量的关系**：标量是0维的张量，向量是1维的张量，矩阵是2维的张量，张量是n维的，与np.array类似*(由于torch和numpy的特殊关系，似乎numpy中array的操作大部分可以在Tensor上实践)*

### torch.Tensor里的dim
与np.array里的axis类似，也是一层一层剥开他的心。

很多人说 *在二维数组里dim/axis=0代表列，dim/axis=1代表行* ，其实这是对函数里dim/axis的**误解**，是一种反常的思路，会给今后的工作带来记忆上的麻烦。
```
import torch
a = torch.rand((3,4))
print(a)
print(a.size())
```
构建的一个随机张量：
```
tensor([[0.1627, 0.0867, 0.5219, 0.5004],
        [0.0758, 0.3883, 0.0479, 0.9100],
        [0.8907, 0.5332, 0.5093, 0.2759]])
torch.Size([3, 4])
```
### 以torch.argmax()为例：
先测试dim=0
```
b= torch.argmax(a,dim=0)
c= torch.argmax(a,dim=0,keepdim=True)
print("dim=0",b)
print("dim=0",b.size())
print("dim=0,keepdim",c)
print("dim=0,keepdim",c.size())
```
输出结果：
```
dim=0 tensor([2, 0, 2, 2])
dim=0 torch.Size([4])
dim=0,keepdim tensor([[2, 0, 2, 2]])
dim=0,keepdim torch.Size([1, 4])
```
输出结果表明，dim=0的意思是:对最外层的块(矩阵里就是行了)做比较
例如，我拿起```a[0]```，也就是```[0.1627, 0.0867, 0.5219, 0.5004]```上的每一个元素，与a[1],a[2]，也就是```[0.0758, 0.3883, 0.0479, 0.9100]```和```[0.8907, 0.5332, 0.5093, 0.2759]```上的每一个元素去比，返回一个```tensor([[2, 2, 0, 1]])```size(1,4),也就是4个行索引。看起来是找每一列中最大的那一行，但归根结底是对行进行操作。由于```keepdim=False```，所以把新的tensor里dim=0这一行的壳给去了，变成了```dim=0 tensor([2, 0, 2, 2])```size(4)

同样对dim=1 进行测试：
```
b= torch.argmax(a,dim=1)
c= torch.argmax(a,dim=1,keepdim=True)
print("dim=1",b)
print("dim=1",b.size())
print("dim=1,keepdim",c)
print("dim=1,keepdim",c.size())
```
输出结果
```
dim=1 tensor([1, 0, 3])
dim=1 torch.Size([3])
dim=1,keepdim tensor([[1],
        [0],
        [3]])
dim=1,keepdim torch.Size([3, 1])
```

### 以torch.stack()为例

```
import torch
a = torch.zeros((3,4))
b = torch.ones((3,4))
print(a)
print(a.size())
print(b)
print(b.size())
```
初始化两个向量：
```
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
torch.Size([3, 4])
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
torch.Size([3, 4])
```
dim = 0
```
c = torch.stack((a,b),dim=0)
print(c)
print(c.size())
```
涉及到拼接，dim在原张量的基础上扩张了1维，所以这里的dim = 0 是指把a作为1块，b作为1块，拼接出一个```Size([2, 3, 4])```的憎恶（误）。```size[0]=2``` 是因为有俩张量在拼接(a和b)
```
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
torch.Size([2, 3, 4])
```
dim = 1，同样地，维度扩增了1，dim=1 表示在原张量的dim = 0的子张量(对a来说就是```[0., 0., 0., 0.]```)拿出来拼接，所以得到了Size([3, 2, 4])
```
tensor([[[0., 0., 0., 0.],
         [1., 1., 1., 1.]],

        [[0., 0., 0., 0.],
         [1., 1., 1., 1.]],

        [[0., 0., 0., 0.],
         [1., 1., 1., 1.]]])
torch.Size([3, 2, 4])
```
dim = 2，同理，得到一个Size([3, 4, 2])
```
tensor([[[0., 1.],
         [0., 1.],
         [0., 1.],
         [0., 1.]],

        [[0., 1.],
         [0., 1.],
         [0., 1.],
         [0., 1.]],

        [[0., 1.],
         [0., 1.],
         [0., 1.],
         [0., 1.]]])
torch.Size([3, 4, 2])
```
总之axis=i 就在size的i位置上赋值 len（num_tensors）这里是2个。
