# 使用教程
## 1.筛选数据集
### 1.下载
数据集我们采用 *HumBugDB: A Large-scale Acoustic Mosquito Dataset* 中`所提供的蚊子叫声数据集。

数据集可从此处获取:`https://zenodo.org/record/4904800`
本数据集相应的标注文件可从github上获取:`https://github.com/HumBug-Mosquito/ZooniverseData`

---
### 2.清洗（可选）
由于全部数据集过于庞大，首先于笔者自身电脑容量限制，我们只选取了`humbugdb_neurips_2021_1`这一部分的数据集进行使用。
若不使用全部数据集，可使用dataset_sampler.py 进行清洗
---
### 3.转换
完成数据集的清洗后我们需要将