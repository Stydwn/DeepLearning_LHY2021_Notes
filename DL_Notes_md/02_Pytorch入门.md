# Pytorch入门

### Pytorch/Tensor介绍

利用Pytorch进行DNN的过程：

![image-20211027212655474](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211027212655474.png)

**Tensor**: 高纬度的矩阵

![image-20211027174956185](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211027174956185.png)

### tensor基本操作

```
dim in Pytorch == axis in Numpy
```

**初始化方法：**

```python
#From list / NumPy array
x = torch.tensor([[1, -1], [-1, 1]])
x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))
#Zero tensor
x = torch.zeros([2, 2])
#Unit tensor
x = torch.ones([1, 2, 5])
```

**Squeeze() and unsqueeze()：**

这两个方法分别对应**减少一维**和**增加一维**，注意只有维度为1时才会去掉或增添，里面的参数代表要增减维度的位置。

```python
#input[1]:
import torch
import numpy as np
x = torch.zeros([1,2,1])
x = x.squeeze(-1)
print('squeeze(-1):',x.shape)
x = x.unsqueeze(-2)
print('unsqueeze(-2):',x.shape)
```

```python
#output[1]:
squeeze(-1): torch.Size([1, 2])
unsqueeze(-2): torch.Size([1, 1, 2])
```

**transpose(dim0,dim1)**

转置操作，两个参数代表要转置的两维度。

```python
#input[2]
a = torch.tensor([[[0,1,2],[3,4,5]]])
print("a.shape:", a.shape)
print('a:\n',a)
d = a.transpose(0,1)
print("d.shape:", d.shape)
print('d\n',d)
e = a.transpose(2,1)
print("e.shape:", e.shape)
print('e\n',e)
```

```python
#output[2]
a.shape: torch.Size([1, 2, 3])
a:
 tensor([[[0, 1, 2],
         [3, 4, 5]]])
d.shape: torch.Size([2, 1, 3])
d
 tensor([[[0, 1, 2]],

        [[3, 4, 5]]])
e.shape: torch.Size([1, 3, 2])
e
 tensor([[[0, 3],
         [1, 4],
         [2, 5]]])
```

**cat()**

按指定的维度将张量拼接起来

```python
#input[3]
x = torch.zeros(2,2,3)
y = torch.zeros(2,3,3)
z = torch.zeros(2,6,3)
w = torch.cat([x,y,z],1)
print(w.shape)
```

```
#output[3]
torch.Size([2, 11, 3])
```

**+/-/pow()/sum()/mean()**



### How to Calculate Gradient?

**eg:**![image-20211027222916210](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211027222916210.png)

```
第一步：
x = torch.tensor([[1., 0.], [-1., 1.]], 
```

```
第二步：
z = x.pow(2).sum()
```

```
第三步：
z.backward()
```

```python
第四步：
#input[4]
x.grad
#outpur[4]
tensor([[ 2.,  0.],
    	[-2.,  2.]])

```



### 读数据Dataset & Dataloader

Dataset：对原始数据进行读取并进行预处理。

Dataloader：从Dataset中shuffle后的结果中拿取一部分data

![image-20211027223902298](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211027223902298.png)



###   Neural Network Layers（全连接）

```python
nn.Linear(in_features, out_features)
#eg:in_features = 32 out_features=64（输入的最后一维度必须是32）
```

![image-20211027224722185](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211027224722185.png)

x：features W：联想到播放量的例子代表激活函数的个数xfeature的个数

![image-20211027225343242](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211027225343242.png)

### 定义$Loss$

对于回归问题：

```
MSELoss()
```

对于分类问题：

```
CrossEntropyLoss()
```



### 定义Model：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
        	
            nn.Linear(10, 32)  #第一层laye
            
            nn.Sigmoid()       #激活函数
            
            nn.Linear(32, 1)   #第二层layer
        )

    def forward(self, x):
        return self.net(x)     #返回你定义的Model
```

上述代码过程就是在激活函数是$Sigmoid$的条件下先通过第一层layer输出32features在通过一层 layer输出1features。



### 优化步骤**Optimization algorithms** ：

```python
torch.optim.SGD(model.parameters(),learning rate, momentum = 0)
```



### 训练步骤：

```python
dataset = MyDataset(file)  #读数据并初始化
tr_set = DataLoader(dataset, 16, shuffle=True) #取出datase中的部分数据作为训练集
model = MyModel().to(device)  #modle的训练机器
criterion = nn.MSELoss() #损失函数
optimizer = torch.optim.SGD(model.parameters(), 0.1) #优化方法

```

```python
for epoch in range(n_epochs): # 训练的epoch
    	model.train() # 设置model traning状态
    	for x, y in tr_set: # x代表feature y代表label
        	optimizer.zero_grad() # 将gradient设为0，防止上一步未清空的梯度
        	x, y = x.to(device), y.to(device) #移动data到训练机器中
        	pred = model(x) # 该Model输出的结果
        	loss = criterion(pred, y) # 算出该Model的Loss
        	loss.backward() # 算出gradient
        	optimizer.step()# 依据上一步算出的gradient更新Model 
```

验证模型在训练集性能：

```python
model.eval() # 设置模型为evaluation mode
total_loss = 0 # 初始化loss
for x, y in dv_set:
    	x, y = x.to(device), y.to(device)
    	with torch.no_grad():# 设置Model不算gradient
        	pred = model(x) # 该Model输出的结果
        	loss = criterion(pred, y)
    	total_loss += loss.cpu().item() * len(x) # 累加loss
	avg_loss = total_loss / len(dv_set.dataset) # 取均值

```

输入预测结果：

```
model.eval() # 设置模型为evaluation mode
preds = []
for x in tt_set:
    x = x.to(device)
    with torch.no_grad(): # 设置Model不算gradient
        pred = model(x) # 预测结果
        preds.append(pred.cpu()) # 收集预测的结果

```





