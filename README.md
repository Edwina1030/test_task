# test_task
1. Background: test task for getting the chance to join the PMB WS2020_2021
2. Demand: based on a given dataset, use a CNN model, with Python and Pytorch to classify ten kinds of fonts of number
3. Github Link for direct clone: git@github.com:Edwina1030/test_task.git
4. Test on  MacOS 10.15.7 and Ubuntu 12.04

# How to works?
**1. setting virtual environment**
```
pip3 install virtualenv
```
**2. run setup.sh in current category**
```
./setup.sh
```
Requirements.txt has problem about 'dataclasses == 0.7'. For better compatibility, ithas been manually modified as 'dataclasses == 0.6'

**3. run dataset_split.py**
```
python3 dataset_split.py
```
It could automatically split the data folder with a ratio of 0.8:0.2. A new folder named as target_data will be created, which includes a train_val subfolder and test subfolder.
- train_val subfolder includes training and validation dataset. A new txt.file named as train_val will be created, which contains names and labels of training and validation dataset.
- test subfolder includes test dataset. A new txt.file named as test.txt will be created, which includes names and labels of the test dataset.

**4. run train_script.py**
```
python3 train_script.py
```
- dataloader.py and model.py will be automatically imported 
- Based on the previous implementation, parameters with relatively better performance haven been decided, like LR, batch size, epoch, optimizer etc. 
- During training, train_val folder will be randomly split into training and validation dataset with ratio 0.8:0.2
- After training, best model learning_curve.jpg will be saved in the current category. (when not in PyCharm IDE, can not direct plot learning curve)
**5. run evaluation_script.py**
```
python3 evaluation_script.py
```
saved best model will be loaded. After testing, a confusion matrix will be saved, and necessary evaluation standards will be print out


# To-Do List 12.11.2020 updated
1. with Pytorch build a model
* [x] 4 layers CNN model
* [x] built-in MNIST dataset from PyTorch
* [x] training works
* [x] running in CoLab
* [x] can classify 10 classes 

2. analyze the given data set
* [x] read txt file 
* [x] how many images of each class?
* [x] load dataset into Google Drive
* [x] build my Dataset class, take care about “；” in txt file
* [x] plot
* [x] balance the training data set
* [x] spilt main.py, run codes in my own laptop
* [x] split source dataset into subsets(train_val, test)

3. extend the previous model to support the given data set
* [x] change model 
* [x] confusion matrix
* [x] change the way to save model, not the last model. At the point where the loss function value of the validation set changes from falling to rising (i.e. the minimum), the model has the best generalization ability.
* [x] Analyse standards, F1-Score, Precise, etc
* [x] normalise Confusion Matrix
* [x] loss function
* [x] activation function
* [x] dropout, make no great sense in this dataset
* [x] batch normalisation, make no great sense in this dataset
* [x] optimizer, may use dynamic learning rate, finally using Adam
* [x] learning rate, using dichotomy, finally decided LR = 0.001
* [x] batch size 64
* [x] epochs, in 10-30 epochs, 15 epochs are enough
* [x] adjust space between subplots 

4. for cross-platform running
* [x] modify dataset_split.py, which can automatically split source dataset
* [x] according to ratio split training and validation dataset


