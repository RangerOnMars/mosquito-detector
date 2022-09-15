# 使用教程
## 1.筛选数据集
### 1.下载
数据集我们采用 *HUMBUG ZOONIVERSE: A CROWD-SOURCED ACOUSTIC MOSQUITO DATASET* 中`1所提供的蚊子叫声数据集。

在终端执行如下命令以下载数据集
```
wget http://humbug.ac.uk/public/Zooniverse_audio_1sec.zip
```
本数据集相应的标注文件可从github上获取:`https://github.com/HumBug-Mosquito/ZooniverseData`

---
### 2.清洗
修改`dataset_sampler.py`，将相关路径替换为自己的路径，执行

---
### 3.转换
得到