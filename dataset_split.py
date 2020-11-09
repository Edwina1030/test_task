import os
import shutil
import numpy as np
import pandas as pd

txt2array = np.loadtxt('target.txt', dtype=str, delimiter=';')
print('Content of source txt file:', txt2array)
print('Shape of source txt file:', txt2array.shape)
print('===' * 20)

train_val_len = int(txt2array.shape[0] * 0.8)
train_val_array = txt2array[0:train_val_len, :]
print('Content of train and val txt file', train_val_array)
print(train_val_array.shape)
print('===' * 20)

train_val_label = train_val_array[:, 1]
print('Label of train and val images', train_val_label)
train_val_label = pd.Series(train_val_label)
train_val_label_count = train_val_label.value_counts()
train_val_label_count.sort_index(inplace=True)
print('statistic of train and val image labels')
print(train_val_label_count)
print('===' * 20)

test_len = txt2array.shape[0] - train_val_len
test_array = txt2array[train_val_len:, :]
print(test_array.shape)
print('===' * 20)

test_label = test_array[:, 1]
test_label = pd.Series(test_label)
test_label_count = test_label.value_counts()
test_label_count.sort_index(inplace=True)
print('statistic of test image labels')
print(test_label_count)
print('===' * 20)

np.savetxt('train.txt', train_val_array, fmt="%s;%s", delimiter=";")
np.savetxt('test.txt', test_array, fmt="%s;%s", delimiter=";")


def copy(srcpath, dstpath, filename):
    if not os.path.exists(srcpath):
        print("srcpath not exist!")
    if not os.path.exists(dstpath):
        print("dstpath not exist!")
    for root, dirs, files in os.walk(srcpath, True):
        if filename in files:
            shutil.copy(os.path.join(root, filename), dstpath)
        else:
            print(filename)


for filename in train_val_array[:, 0]:
    copy('./data', './target_data/train_val', filename)
for filename in test_array[:, 0]:
    copy('./data', './target_data/test', filename)
