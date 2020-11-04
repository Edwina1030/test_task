import numpy as np
import pandas as pd
# load txt.file as array
txt2array = np.loadtxt('/Users/tianhengling/Documents/7. Master Studium/17 Masterprojekt II/Test Task/mnist/target.txt', dtype=str, delimiter=';')
print('the content of txt file:',txt2array)
print('the shape of text file:',txt2array.shape)
print('==='*20)

image_name = txt2array[:,0]
print('name of all images:',image_name)

image_label = txt2array[:,1]
print('label of all images:',image_label)

print('==='*20)
image_label = pd.Series(image_label)
label_count = image_label.value_counts() # transfer type
label_count.sort_index(inplace=True)
print('satistic of image labels')
print(label_count)

# manuel split
size = 1797 // 5
image_label1 = image_label[0:size-1]
image_label2 = image_label[size:2*size-1]
image_label3 = image_label[2*size:3*size-1]
image_label4 = image_label[3*size:4*size-1]
image_label5 = image_label[4*size:]

print('==='*20)
image_label1 = pd.Series(image_label1)
label_count1 = image_label1.value_counts() # transfer type
label_count1.sort_index(inplace=True)
print('satistic of image label 1')
print(label_count1)

print('==='*20)
image_label2 = pd.Series(image_label2)
label_count2 = image_label2.value_counts() # transfer type
label_count2.sort_index(inplace=True)
print('satistic of image label 2')
print(label_count2)

print('==='*20)
image_label3 = pd.Series(image_label3)
label_count3 = image_label3.value_counts() # transfer type
label_count3.sort_index(inplace=True)
print('satistic of image label 3')
print(label_count3)

print('==='*20)
image_label4 = pd.Series(image_label4)
label_count4 = image_label4.value_counts() # transfer type
label_count4.sort_index(inplace=True)
print('satistic of image label 4')
print(label_count4)

print('==='*20)
image_label5 = pd.Series(image_label5)
label_count5 = image_label5.value_counts() # transfer type
label_count5.sort_index(inplace=True)
print('satistic of image label 5')
print(label_count5)