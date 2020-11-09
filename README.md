# test_task
1. Background: test task for getting the chance to join the PMB WS2020_2021
2. Demand: based on given dataset, use a CNN model, with Python and Pytorch to classify 10 kinds of  fonts of number

# list of all files
1. data folder 
given folder of source data
2. target.txt
given txt file which including name of images and its labels
3. target_data folder 
after analyzing target.txt, according to the sequence in text.file, images of each label distribute equally, so create a dataset_split.py to  copy first 80\% images as train and val dataset, latter 20\% images as test dataset, which are copied to train_val and test folder in target data folder
4. train_val.txt, test_txt
because of 3. target.txt also be split into train_val.txt and test_txt
5. dataloader.py
its's a given dataset, can't call built-in MNIST dataset
should self-defined a dataset class
6. model.py
define a model function, 4 layers and 1 fully connected layer CNN model
7. train_script.py + model.pkl+learning_curve.jpg
feed data into model, including some initialization setting for training and validation
the last model with be saved
using matplotlib pot learning curve and save it as jpg
8. evaluation_script.py + confusion_matrix.jpg
load saved model, test it on test dataset
print out loss and acc on test dataset
give a confusion matrix

## To Do List 09.11.2020 updated
1. with Pytorch build a model
* [x] 4 layers CNN model
* [x] built-in MNIST dataset from PyTorch
* [x] trainning works
* [x] running in CoLab
* [x] can classify 10 classes 

2. analyze the given data set
* [x] read text file 
* [x] how many images of each classes?
* [x] load dataset into google Drive
* [x] build my Dataset class, take care about “；” in text file
* [x] plot
* [x] balance the training data set
* [x] spilt main.py, run codes in my own laptop
* [x] split source dataset into subsets(train_val, test)

3. extend the previous model to support the given data set
* [x] running in local computer
* [x] change model 
* [x] confusion matrix
* [ ] change the way to save model, not the last model, to finde a model both good at loss and acc
* [ ] Analyse standards, F1-Score, Precise, etc
* [ ] loss function
* [ ] activation function
* [ ] dropout
* [ ] batch normalisation
* [ ] optimizer 
* [ ] batch size
* [ ] learning rate


