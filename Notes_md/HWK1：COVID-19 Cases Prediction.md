# COVID-19 Cases Prediction 

### 优化步骤

1. 因为数据较为离散，对数据集采用了$\texttt{Standard score}$标准化方法。其他标准化方法参考https://www.cnblogs.com/chester-cs/p/12679316.html
2. 因为特征值太多，利用$sklearn$中的$SelectKBest$的算法选取了和$label$最相关的15个特征
3. 训练时使用了Adam优化器，可以对历史梯度的震荡情况和过滤震荡后的真实历史梯度对变量进行更新
4. 使用$Dropout$随机丢弃数据集防止过拟合
5. 为了防止过拟合加入了$regulation$项及$L2$正则化
6. 使用$BatchNorm1d$对参数进行归一化处理加快训练速率
7. 因为训练数据太少，所以采用 训练集 == 验证集 的方法

### 训练结果

![image-20211029232701177](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211029232701177.png)

![image-20211029232709919](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211029232709919.png)

**在Kaggle上提交勉强能过Public Strong Baseline，差一点点过Private Strong baseline还需要更深层次的优化  :(**

### 踩坑

1. $Model$中的$forward$方法中返回的模型$size$是[32,1],而输入的$size$是[32]，要用suqeeze降维。

   问题原因：线性回归网络的输出是1维，而在读取target数据时，默认读取为了一维向量，而预测的结果是tensor，是在一维的基础上unsuqeeze了batch维度得到的，而在计算MSEloss时候，维度不同时计算loss可能导致错误从而导致训练提前终止。

2. 报错信息 Traceback(most recent call last) + object of type ' ‘ has no len()

   问题原因：继承Dataset的类重写方法时一个地方没有注意缩进

### Code

```python
# 选取特征方法
def select_features(path):
    data = pd.read_csv(path)
    x = data[data.columns[1:94]]
    x1 = data[data.columns[1:94]]
    y = data[data.columns[94]]
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))
    x[:, 40:] = (x[:, 40:] - x[:, 40:].mean(dim=0)) / x[:, 40:].std(dim=0) # 标准化
    bestfeatures = SelectKBest(score_func=f_regression, k=5)
    fit = bestfeatures.fit(x, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x1.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1) # 利用pandas进行输出查看
    featureScores.columns = ['Specs', 'Score'] 
    print(featureScores.nlargest(15, 'Score'))
```

```python
# Dataset
class Coid19_dataset(Dataset):
    def __init__(self,path,mode='train'):
        super().__init__()
        self.mode = mode

        #读csv文件
        with open(path) as file:
            data_csv = list(csv.reader(file))
            data = np.array(data_csv[1:])[:,1:].astype(float)
        feats = [75, 57, 42, 60, 78, 43, 61, 79, 40, 58, 76, 41, 59, 77]  # 上面挑选的最优特征
        if mode == 'test':
            data = data[:,feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:,-1]
            data = data[:,feats]

            indices = [i for i in range(data.shape[0])]


            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0)) / self.data[:, 40:].std(dim=0) #标准化
        self.dim = self.data.shape[1] #获取数据集的feature数量

    def __getitem__(self, item):  
        if self.mode == 'train' or self.mode == 'dev':  
            return self.data[item], self.target[item]
        else:
            return self.data[item]  

    def __len__(self):  
        return len(self.data)

```

```python
# Dataloader
ef coid19_dataloader(path,mode,batch_size,n_jobs=0):
    coid19_dataset = Coid19_dataset(path,mode)
    data_loader = DataLoader(coid19_dataset,batch_size,
                            shuffle=(mode == 'train'),
                            drop_last=False,
                            num_workers=n_jobs,
                            pin_memory=False
                            )
    # print(mode,'dataload sucess!!!')
    return data_loader
```

```python
# Model
class MyModel(nn.Module):
    def __init__(self,input_dim):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),

            nn.BatchNorm1d(32), # BN加速模型训练速率
            nn.Dropout(p=0.2), # Dropout减小过拟合

            nn.LeakyReLU(),

            nn.Linear(32,1)
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self,x):
        return self.net(x).squeeze(1)

    def cal_loss(self,pred,target,model): #计算loss
        regularization_loss = 0
        for param in model.parameters():# L2正则化 减少过拟合
            regularization_loss += torch.sum(param**2)
        return self.criterion(pred,target) + 0.00075 * regularization_loss
```

```python
# 当前模型下验证集的Loss方法
def dev(model, dev_data):
    model.eval()  # 设置模式
    total_loss = []

    for x, y in dev_data:  # 得到当前模型下验证集的Loss方法
        pred = model(x)
        dev_loss = model.cal_loss(pred, y,model)
        total_loss.append(dev_loss)

    return sum(total_loss) / len(total_loss)  
```

```python
# train
def train(model, train_data, dev_data):
    max_epoch = 11000  # 至多训练次数
    epoch = 1

    optimizer = getattr(torch.optim, 'Adam')(
        model.parameters()) # Adam优化器

    train_loss = []  # 存储训练集的Loss
    dev_loss = []  # 存储测试集的Loss
    min_mse = 1000
    break_flag = 0
    while epoch < max_epoch:
        model.train()  # 设置模式
        for x, y in train_data:  # x，y 每次包含一个batch_size的样本
            optimizer.zero_grad()  # 每次必须先将梯度清零
            pred = model(x)
            loss = model.cal_loss(pred, y,model)  # 计算Loss
            train_loss.append(loss.detach())
            loss.backward()  # 计算梯度
            optimizer.step()  # 模型的参数更新

        dev_mse = dev(model, dev_data)
        if dev_mse < min_mse:  # 如果测试验证集的Loss比上一次小，则存储当前模型
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))

            # 存储当前最好的模型，此处需要导入os库，创建相应的目录
            torch.save(model.state_dict(), 'my_models/mymodel.pth')

            break_flag = 0
        else:
            break_flag += 1

        dev_loss.append(dev_mse.detach())

        if break_flag > 500:  # 如果连续500个epoch，Loss都没有下降,结束训练
            break

        epoch += 1
    return train_loss, dev_loss
```



 ```python
 # 在测试集上进行预测并保存到本地文件
 def test(model, test_data):
     model.eval()  
     preds = []
     for x in test_data:  
         pred = model(x)
         preds.append(pred.detach().cpu())
     preds = torch.cat(preds, dim=0).numpy()
     return preds
 def save_pred(preds, file):
     print('Saving results to {}'.format(file))
     with open(file, 'w') as fp:
         writer = csv.writer(fp)
         writer.writerow(['id', 'tested_positive'])
         for i, p in enumerate(preds):  
             writer.writerow([i, p])
 ```

```python
# train_loss 和 dev_loss 随epoch的变化图
def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(dev_loss)]
    plt.figure(1, figsize=(6, 4))
    plt.plot(x_1, train_loss, c='tab:red', label='train')
    plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()
```



```python
# 训练的模型和数据点的拟合图
def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    plt.figure(2, figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
```



```python
#执行函数

# 设置存储模型的目录
os.makedirs('my_models', exist_ok=True)
# select_features('covid.train.csv')
# 加载数据
train_data = coid19_dataloader('covid.train.csv', 'train', batch_size=150)
dev_data = coid19_dataloader('covid.train.csv', 'dev', batch_size=200)
test_data = coid19_dataloader('covid.test.csv', 'test', batch_size=200)

# 设置模型与训练
mymodel = MyModel(train_data.dataset.dim)
train_loss, dev_loss = train(mymodel, train_data, dev_data)
plot_learning_curve(train_loss, dev_loss, title='deep model')
del mymodel

# 加载最好的模型进行预测
model = MyModel(train_data.dataset.dim)
ckpt = torch.load('my_models/mymodel.pth', map_location='cpu')  # 加载最好的模型
model.load_state_dict(ckpt)
plot_pred(dev_data, model, 'cpu')
preds = test(model, test_data)

# 储存预测结果
save_pred(preds, 'mypred.csv')

```

