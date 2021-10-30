# Batch & Momentum

### Batch的大小对$Model$的影响

- Batch比较大时训练的$Model$比较powerful
- Batch比较小时训练的$Model$比较noisy

**原因：**Batch比较小时每次update的数据多，对模型的拟合能力更强。



### Batch的大小对训练$Model$时间的影响

由于GPU并行计算的原因， 随batch的增加对update和epoch时间的影响如图。

![image-20211030233443841](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211030233443841.png)



### Batch的大小对训练$Model$性能的影响

![image-20211030233630788](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211030233630788.png)

可以看出$Model$使用Large Batch比Small Bath的效果更差。

**原因：**Large Batch会陷入局部最优的情况，也就是说在训练过程中梯度消失的问题。Small Batch每次的Loss Function都略有不同，如果在一个Batch上出现梯度消失的情况，虽然这次没有更新参数，但下一个Batch的数据改变了再次出现梯度消失的概率几乎为0。

![image-20211030234718556](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211030234718556.png)

###  Momentum

由于Gradient Descent遇到critica point时会出现梯度消失的情况，所以我们可以考虑上一次Gradient Descent的方向和这次微分后的方向来决定train的方向，类似于物理中的惯性

**总结：Update不仅考虑了当前的gradient还考虑了过去所有Gradient的总和**



