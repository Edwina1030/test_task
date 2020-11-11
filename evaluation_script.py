import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import TorchDataset
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

test_filename = "./test.txt"
test_dir = "./target_data/test"
test_set = TorchDataset(filename=test_filename, image_dir=test_dir, repeat=1)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

model = torch.load('./model.pkl')
criterion = nn.CrossEntropyLoss()

test_loss = 0
test_acc = 0

for img, label in test_loader:
    img = Variable(img)
    label = Variable(label)

    out = model(img)
    loss = criterion(out, label.reshape(-1))

    test_loss += loss.item()

    _, pred = out.max(1)
    num_correct = (pred == label.reshape(-1)).sum().item()
    acc = num_correct / img.shape[0]
    test_acc += acc
print('Model accuracy on test dataset:', round(test_acc / len(test_loader),3))
print('Model loss on test dataset:', round(test_loss / len(test_loader),3))

y_true = label.reshape(-1)
y_pred = pred
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.round(cm,2)
print(cm)

xtick=['0','1','2','3','4','5','6','7','8','9']
ytick=['0', '1','2','3','4','5','6','7','8','9']

sn.heatmap(cm,fmt='g', cmap='Blues', annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick)
plt.tight_layout()
plt.title('Confusion Matrix of 10 kinds of fonts')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("./confusion_matrix.jpg")
plt.show()

print("===="*6,'report',"===="*6)
macro_precision = precision_score(y_true, y_pred,average='macro')
print('macro precision = ', round(macro_precision,3))
micro_precision = precision_score(y_true, y_pred,average='micro')
print('micro precision = ', round(micro_precision,3))
weighted_precision = precision_score(y_true, y_pred,average='weighted')
print('weighted precision = ', round(weighted_precision,3))
print("----"*15)

accuracy = accuracy_score(y_true, y_pred)
print('accuracy = ', round(accuracy,3))
print("----"*15)

macro_recall = recall_score(y_true, y_pred, average='macro')
print('macro recall = ', round(macro_recall, 3))
micro_recall = recall_score(y_true, y_pred, average='micro')
print('micro recall = ', round(micro_recall,3))
weighted_recall = recall_score(y_true, y_pred, average='weighted')
print('weighted recall = ', round(weighted_recall,3))
print("----"*15)
macro_f1_score = f1_score(y_true, y_pred, average='macro')
print('macro F1 score = ', round(macro_f1_score,3))
micro_f1_score = f1_score(y_true, y_pred, average='micro')
print('micro F1 score = ', round(micro_f1_score,3))
weighted_f1_score = f1_score(y_true, y_pred, average='weighted')
print('weighted F1 score = ', round(weighted_f1_score,3))