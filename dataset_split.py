import os
import shutil
import numpy as np
import pandas as pd

def copy(srcpath, dstpath, filename):
    for root, dirs, files in os.walk(srcpath, True):
        if filename in files:
            shutil.copy(os.path.join(root, filename), dstpath)
        else:
            print(filename)

srcpath = '/usr/src/data' # container中路径，已经被映射到local path中的data file
if not os.path.isdir(srcpath):
    print('Warning: Source dataset not exists!')
    print('Please add a dataset folder named as data!')

print('=='*5,'Start analyzing txt file of source dataset','=='*5)
txt2array = np.loadtxt('target.txt', dtype=str, delimiter=';')
print(txt2array)
print(txt2array.shape)

print('=='*2,'80% of source dataset wiill be split as training and validation dataset','=='*2)
train_val_len = int(txt2array.shape[0] * 0.8)
train_val_array = txt2array[0:train_val_len, :]
print(train_val_array)
print(train_val_array.shape)

train_val_label = train_val_array[:, 1]
print('Successffully get labels of train and validation images!')
print(train_val_label)
train_val_label = pd.Series(train_val_label)
train_val_label_count = train_val_label.value_counts()
train_val_label_count.sort_index(inplace=True)
print('Check the type and quantity of labels!')
print(train_val_label_count)
np.savetxt('train_val.txt', train_val_array, fmt="%s;%s", delimiter=";")
print('Successfully save txt file of training and validation dataset!')

train_val_path = '/usr/src/target_data/train_val' # container中路径，已经被映射到local path中的data file
if not os.path.isdir(train_val_path):
    os.makedirs(train_val_path)
    print('Training and validation dataset is successfully created!')
else:
    print('Training and Validation dataset: The original dataset division has been completed!')
for filename in train_val_array[:, 0]:
    copy(srcpath, train_val_path, filename)


print('=='*2,'Rest of source dataset will be split as test dataset','=='*2)
test_len = txt2array.shape[0] - train_val_len
test_array = txt2array[train_val_len:, :]
print(test_array)
print(test_array.shape)

test_label = test_array[:, 1]
print('Successffully get labels of test images!')
print(test_label)
test_label = pd.Series(test_label)
test_label_count = test_label.value_counts()
test_label_count.sort_index(inplace=True)
print('Check the type and quantity of labels!')
print(test_label_count)
np.savetxt('test.txt', test_array, fmt="%s;%s", delimiter=";")
print('Successfully save txt file of test dataset!')

test_path = '/usr/src/target_data/test' # container中路径，已经被映射到local path中的data file
print(test_path)
if not os.path.isdir(test_path):
    os.makedirs(test_path)
    print('Training and validation dataset is successfully created!')
else:
    print('Test dataset: The original dataset division has been completed')

for filename in test_array[:, 0]:
    copy(srcpath, test_path , filename)

print('=='*5,'The automatic split of source dataset is finished','=='*5)
print('Split data are in target_data folder')