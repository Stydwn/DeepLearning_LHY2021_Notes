﻿## 1.使用指令`conda info --envs`查询安装环境时未出现`pytorch`

**解决方案**：在环境配置文件`.condarc`（一般在user/用户名 目录下）重新设置镜像源：

```bash
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
ssl_verify: true
```
然后重新设置环境屋：

```bash
conda create -n xxxx python=xx
```
查询环境：

```bash
conda info --envs
```

![](https://img-blog.csdnimg.cn/2a9524a6871d4fb785541e03011e47e7.png)

## 2.`import pytorch` 出现错误

```bash
File "<stdin>", line 1
    impot pytorch
          ^
SyntaxError: invalid syntax
```

**解决方案**：在`Anaconda Prompt`下进行`import pytorch`
![在这里插入图片描述](https://img-blog.csdnimg.cn/8e1f4702431143e48736c88d5bf0e33d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAc3R5ZHdu,size_20,color_FFFFFF,t_70,g_se,x_16)

## 3.`torch.cuda.is_available`验证pytorch是否可以使用GPU 出现错误`<function is_available at 0x000002A0B66DACA0>`

**解决方案**：指令改成`torch.cuda.is_available()`
![在这里插入图片描述](https://img-blog.csdnimg.cn/98279508afdc460182d2208684787c90.png)

## 4.`import torch`报错没有该package

**解决方案**：File ---> Settings ---> 
![在这里插入图片描述](https://img-blog.csdnimg.cn/8e597bcd07fe49ba8574bcd65e940ddd.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAc3R5ZHdu,size_20,color_FFFFFF,t_70,g_se,x_16)
改为**pytorch**文件夹下的**python.exe**

## 测试：

测试一下成功解决：
![在这里插入图片描述](https://img-blog.csdnimg.cn/a3769134dda347f287cc42fa3850c052.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAc3R5ZHdu,size_20,color_FFFFFF,t_70,g_se,x_16)

