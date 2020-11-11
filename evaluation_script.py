import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import TorchDataset
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
import seaborn as sn
import matplotlib.pyplot as plt

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
print('Model accuracy on test dataset:', test_acc / len(test_loader))
print('Model loss on test dataset:', test_loss / len(test_loader))

y_true = label.reshape(-1)
y_pred = pred
cm = confusion_matrix(y_true, y_pred)

xtick=['1','2','3','4','5','6','7','8','9','10']
ytick=['1','2','3','4','5','6','7','8','9','10']

sn.heatmap(cm,fmt='g', cmap='Blues', annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick)
plt.tight_layout()
plt.title('Confusion Matrix of 10 kinds of fonts')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("./confusion_matrix.jpg")
plt.show()

print("===="*6,'report',"===="*6)
macro_precision = precision_score(y_true, y_pred,average='macro')
print('macro precision = ', macro_precision)
micro_precision = precision_score(y_true, y_pred,average='micro')
print('micro precision = ', micro_precision)
weighted_precision = precision_score(y_true, y_pred,average='weighted')
print('weighted precision = ', weighted_precision)
print("----"*15)
accuracy = accuracy_score(y_true, y_pred)
print('accuracy = ', accuracy)
print("----"*15)
macro_recall = recall_score(y_true, y_pred, average='macro')
print('macro recall = ', macro_recall)
micro_recall = recall_score(y_true, y_pred, average='micro')
print('micro recall = ', micro_recall)
weighted_recall = recall_score(y_true, y_pred, average='weighted')
print('weighted recall = ', weighted_recall)
print("----"*15)
macro_f1_score = f1_score(y_true, y_pred, average='macro')
print('macro F1 score = ', macro_f1_score)
micro_f1_score = f1_score(y_true, y_pred, average='micro')
print('micro F1 score = ', micro_f1_score)
weighted_f1_score = f1_score(y_true, y_pred, average='weighted')
print('weighted F1 score = ', weighted_f1_score)