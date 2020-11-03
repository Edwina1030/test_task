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
