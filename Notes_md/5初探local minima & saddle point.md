# 初探$\texttt{local minima}$  & $\texttt{saddle point}$ 

### 产生原因

函数在某处的微分值为0及$\texttt{critical point}$导致训练提前终止得到了很大的Loss。$\texttt{critical point}$包括$\texttt{local minima}$和$\texttt{saddle point}$。

![image-20211030191426552](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211030191426552.png)



### 判断$\texttt{local minima}$  & $\texttt{saddle point}$ & $\texttt{local maxima}$

用到的数学知识：

1. 泰勒展开
2. 黑塞矩阵判断多元函数极值
3. 二次型理论
4. 矩阵特征值



在$θ^′$一处的Loss可以用$θ$(变量)进行泰勒展开表示：
$$
𝐿(𝜽)≈𝐿(𝜽^′ )+(𝜽−𝜽^′ )^𝑇 {\color{green}𝒈}+1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )
$$


**$\color{green}g$** 是一个向量，就是Loss Function在$𝜽_i$处的偏微分
$$
{\color{green}𝒈}=𝛻𝐿(𝜽^′ )  \qquad\qquad{\color{green}𝒈_𝑖}=𝜕𝐿(𝜽^′ )/(𝜕𝜽_𝑖 )
$$


Hessian $\color{red}H$  是一个矩阵，就是Loss Function在$𝜽$处的二次微分
$$
{\color{red}𝐻_{ij}}=𝜕^2/(𝜕𝜽_𝑖 𝜕𝜽_𝑗 ) 𝐿(𝜽^′ )
$$
$1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )$ 和 $(𝜽−𝜽^′ )^𝑇{\color{green}𝒈}$ 代表的含义如图所示：

![image-20211030195216673](C:\Users\stydwn\AppData\Roaming\Typora\typora-user-images\image-20211030195216673.png)

当出现$\texttt{critical point}$时微分值为0及泰勒展开式中$(𝜽−𝜽^′ )^𝑇 {\color{green}𝒈}$项为0，我们就可以根据$1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )$ 项判断是local minima还是laddle point。方法：爆搜所有$𝜽$。

- 取任意的 $𝜽$ 如果$1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )>0$可以推出$L(θ)>L(θ^′)$所以这是个**Local minima**
- 取任意的 $𝜽$ 如果$1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )<0$可以推出$L(θ)<L(θ^′)$所以这是个**Local maxima**
- 取任意的 $𝜽$ 如果$1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )><0$可以推出$L(θ)><L(θ^′)$所以这是个**Saddle point**

上述的步骤我们需要穷举 $𝜽$ 计算过于繁琐，根据二次型理论我们可以求出**黑塞矩阵的特征值**：

- 所有特征值都大于0可以推出是**Local minima**
- 所有特征值都小于于0可以推出是**Local maxima**
- 特征值有正有负可以推出是**Saddle point**



### 遇到Saddle point更新

此时 Loss 定义为:
$$
𝐿(𝜽)≈𝐿(𝜽^′ )+1/2 (𝜽−𝜽^′ )^𝑇 {\color{red}𝐻}(𝜽−𝜽^′ )
$$
设${\color{red}𝐻}$的一个特征向量为$u$，令$𝜽−𝜽^′=u$,带入上式末项得到$u^T{\color{red}𝐻}u=u^T (λu)=λ‖u‖^2$整理得到  
$$
𝐿(𝜽)≈𝐿(𝜽^′)+1/2λ‖u‖^2\qquad(λ代表该特征向量的特征值)
$$


显而易见当$λ$为负数时可以推出$L(θ)<L(θ^′ )$，很显然θ处的Loss更小。及由上面的假设可以得到$θ=θ^′+u$，所以更新的方向就是$u$的方向！

**总结成一句话：如果遇到Saddle point更新的方向就是负特征值的特征向量方向！！！**

（在模型训练中更新其实不用这个方法，此例子说明的就是遇到Saddle point有很多方法去解决）

### 小结

通过大量数据可知Local minima是一个假问题，我们遇到的百分之99.99%都是Saddle point的问题。

