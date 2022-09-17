# @Time    : 2022.9.15
# @Author  : RangerOnMars
# @File    : dataset_sampler.py

from sympy import false
from tqdm import tqdm
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

train_txt = "train.txt"
test_txt = "test.txt"
dataset_csv = "data.csv"
data_dir = "gaf"
labels = [["audio","background"]
        ,["mosquito"]]

train_data = []
target_data = []
for root, dirs, files in os.walk(data_dir, topdown=False):
    for file in tqdm(files):
        for i in range(len(labels)):
            if labels[i].count(root.split("/")[-1]) != 0:
                train_data.append(os.path.abspath((os.path.join(root, file))))
                target_data.append(i)
            else:
                continue
print("="*50)
print("Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.3)

print("="*50)
print("Saving Result...")

with open(train_txt,"w") as file:
    for x, y in zip (X_train, y_train):
        file.write(x + " " + str(y) + "\n")
with open(test_txt,"w") as file:
    for x, y in zip (X_test, y_test):
        file.write(x + " " + str(y) + "\n")


print("="*50)
print("Done")