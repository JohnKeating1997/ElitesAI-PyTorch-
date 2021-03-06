# 循环神经网络
## 循环神经网络原理
![](https://github.com/JohnKeating1997/ElitesAI-PyTorch-/blob/master/RNN%E5%85%AC%E5%BC%8F1.PNG)
![RNN模型示意图](https://github.com/JohnKeating1997/ElitesAI-PyTorch-/blob/master/q5jkm0v44i.png)
## 使用的包
```
import torch
import torch.nn as nn
import time
import math
import sys
sys.path.append("/home/kesci/input")
import d2l_jay9460 as d2l         # 这是自定义的包
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## one-hot向量
索引位置为1，其他位置为0的向量，向量长度等于词典大小。

![onehot 批量处理](https://github.com/JohnKeating1997/ElitesAI-PyTorch-/blob/master/RNN%E8%BE%93%E5%85%A5%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)

```
def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result

def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)
```
5 torch.Size([2, 1027])

## 参数初始化
```
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# num_inputs: d
# num_hiddens: h, 隐藏单元的个数是超参数，这里取256
# num_outputs: q

def get_params():
    def _one(shape):
        param = torch.zeros(shape, device=device, dtype=torch.float32)
        nn.init.normal_(param, 0, 0.01)
        return torch.nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))
    return (W_xh, W_hh, b_h, W_hq, b_q)
```
## 定义模型
```
def rnn(inputs, state, params):  #state在rnn里只有一个隐藏层状态
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)   #注意把隐藏层的 H 也返回出来，便于相邻采样时运用

#初始化state（rnn里只有一个H,初始化成batch_size*num_hiddens的0矩阵）
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```
做个简单的测试来观察输出结果的个数（时间步数），以及第一个时间步的输出层输出的形状和隐藏状态的形状。
```
print(X.shape)
print(num_hiddens)
print(vocab_size)
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(inputs), inputs[0].shape)
print(len(outputs), outputs[0].shape)
print(len(state), state[0].shape)
print(len(state_new), state_new[0].shape)
```
输出结果：

torch.Size([2, 5])
256
1027
5 torch.Size([2, 1027])
5 torch.Size([2, 1027])
1 torch.Size([2, 256])
1 torch.Size([2, 256])

## 梯度裁剪
RNN很容易发生梯度衰减，梯度爆炸，这是RNN的反向传播方式决定的，模型参数的梯度是幂的形式，指数是时间步数。随着时间步数的增加，就会发生梯度衰减/爆炸

裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。假设我们把所有模型参数的梯度拼接成一个向量$g$ ，并设裁剪的阈值是$\theta$。裁剪后的梯度$g_{clip}$为：
$$g_{clip}=\min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$
如果$\theta$大于${\|\boldsymbol{g}\|}$，那不用裁剪，$\min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)$取1;  ${\|\boldsymbol{g_{clip}}\|}$ = ${\|\boldsymbol{g}\|}$小于$\theta$

如果$\theta$小于${\|\boldsymbol{g}\|}$，梯度裁剪，${\|\boldsymbol{g_{clip}}\|}=\theta$

总之，裁剪后的 $g_{clip}$ 的$L_2$范数 ${\|\boldsymbol{g_{clip}}\|}$ 不超过$\theta$

```
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
```
## 定义预测方法
以下函数基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符。这个函数稍显复杂，其中我们将循环神经单元rnn设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。
```
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
```
测试一下predict_rnn函数。我们将根据前缀“分开”创作长度为10个字符（不考虑前缀长度）的一段歌词。因为模型参数为随机值，所以预测结果驴唇不对马嘴。
```
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            device, idx_to_char, char_to_idx)
```
'分开濡时食提危踢拆田唱母'

## 困惑度
我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下“softmax回归”一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。
交叉熵：

$$H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)}$$

由于$y_j^{(i)}$只有一个值是1，其他全是0（其实就是one_hot），所以n个样本的平均损失函数可以表示为
$$\ell(\boldsymbol{\Theta}) = -(1/n) \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$$

则困惑度为
$$e^{\ell(\boldsymbol{\Theta})}= \frac{1}{\sqrt[n]{\prod_{i=1}^{n}\hat y_{y^{(i)}}^{(i)}}}$$

最佳情况下，模型总是把**标签类别**的概率预测为1，此时困惑度为1,全对且对得十分确定；
最坏情况下，模型总是把**标签类别**的概率预测为0，此时困惑度为正无穷，全错且错得十分确定；
基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数$n$。

**定义模型训练方法**
跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：

1. 使用困惑度评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。

这里没有仔细看

```
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态 在Tips里有详解
                for s in state:
                    s.detach_()
            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成形状为
            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应
            y = torch.flatten(Y.T)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())
            
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:   #每隔一定的epoch输出预测的信息
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
```

# 使用pytorch简洁实现RNN
## torch.nn.RNN()
使用Pytorch中的**nn.RNN()**来构造循环神经网络。在本节中，我们主要关注nn.RNN的以下几个构造函数参数：

input_size - The number of expected features in the input x
hidden_size – The number of features in the hidden state h
nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
batch_first – If True, then the input and output tensors are provided as (batch_size, num_steps, input_size). **Default: False**

这里的batch_first决定了输入的形状，我们使用默认的参数False，对应的输入形状是 (num_steps, batch_size, input_size)。
```
def forward(input,h_0)  #和自己实现的rnn函数类似，进行前向传播计算
```
官方文档解释：
```
input of shape (num_steps, batch_size, input_size): tensor containing the features of the input sequence.

h_0 of shape (num_layers * num_directions, batch_size, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided. If the RNN is bidirectional(双向神经网络), num_directions should be 2, else it should be 1.
```
forward函数的返回值是：
```
#注意这里output是RNN最后一层（不是最后一步）的隐藏值
output of shape (num_steps, batch_size, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
#h_n是最后一步的所有隐藏层(layers)的值
h_n of shape (num_layers * num_directions, batch_size, hidden_size): tensor containing the hidden state for t = num_steps.
```
## 构造一个rnn_layer
```
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
num_steps, batch_size = 35, 2
X = torch.rand(num_steps, batch_size, vocab_size)
state = None
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)
```

## 构造完整的RNN模型
```
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) #和双向神经网络有关
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)  #最后要有个线性输出层来把隐藏层的结果输出

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size) 这里stack相当于拼接
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state
```
## 定义预测函数
```
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
```
## 使用相邻采样训练函数
```
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state[0].detach_()
                    state[1].detach_()
                else: 
                    state.detach_()
            (output, state) = model(X, state) # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.T)
            l = loss(output, y.long())
            
            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))
```
训练模型并输出
```
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
```
最后出来的效果差强人意
# Tips
1. **torch.tensor.scatter()**
scatter() 和 scatter_() 的作用是一样的，只不过 scatter() 不会直接修改原来的 Tensor，而 scatter_() 会
PyTorch 中，一般函数加**下划线**代表直接在原来的 Tensor 上修改
scatter(dim, index, src) 的参数有 3 个


```
    dim：沿着哪个维度进行索引
    index：用来 scatter 的元素索引
    src：用来替换的源元素，可以是一个标量或一个张量
```

2. **tensor.to(device)**

表示把张量放到device(GPU or CPU)上,再对它进行其他操作
```
X = my_tensor.to(device)
```

my_tensor变量copy一份到device所指定的GPU上去，地址赋给X，之后的运算都在GPU上进行。
```
inputs = to_onehot(X.to(device), vocab_size)
```
X张量放到GPU上，再对它进行to_onehot操作

3. **torch.Variable.detach()**

pytorch 的 Variable 对象中有两个方法，detach和 detach_ : 如**1**所说，区别在于，_表示覆盖原来的

在训练网络的时候可能希望保持前一部分的网络参数不变，只对后**一部分**的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播（也节省了计算开支，因为一直往前求梯度，越深越耗费计算资源）

**detach():**
返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
.这样我们就会继续使用这个新的Variable进行计算，后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播
**detach_()**
将一个Variable从创建它的图中分离，并把它设置成叶子variable.其实就相当于变量之间的关系本来是x -> m -> y,这里的叶子variable是x，但是这个时候对m进行了.detach_()操作,其实就是进行了两个操作：

1. 将m的grad_fn的值设置为None,这样m就不会再与前一个节点x关联，这里的关系就会变成x, m -> y,此时的m就变成了叶子结点。
2. 然后会将m的requires_grad设置为False，这样对y进行backward()时就不会求m的梯度。

其实detach()和detach_()很像，两个的区别就是detach_()是对本身的更改，detach()则是生成了一个新的variable。比如x -> m -> y中如果对m进行detach()，后面如果反悔想还是对原来的计算图进行操作还是可以的。但是如果是进行了detach_()，那么原来的计算图也发生了变化，就不能反悔了。


**官方文档的解释：**
```
返回一个新的从当前图中分离的 Variable。返回的 Variable 永远不会需要梯度

如果 被 detach 的Variable volatile=True， 那么 detach 出来的 volatile 也为 True(没懂)

还有一个注意事项，即：返回的 Variable 和 被 detach 的Variable **指向同一个 tensor**，一个值改变了另外一个也会变
```
```
# y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度
# 第一种方法，y本身没有被阉割， z 得到的只是被阉割的 y 的幻象
y = A(x)
z = B(y.detach())
z.backward()
  
# 第二种方法，直接把y阉了，狼灭......
y = A(x)
y.detach_()
z = B(y)
z.backward()
```
