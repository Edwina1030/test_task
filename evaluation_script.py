import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import TorchDataset
from sklearn.metrics import confusion_matrix
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


cm = confusion_matrix(label.reshape(-1), pred)

xtick=['1','2','3','4','5','6','7','8','9','10']
ytick=['1','2','3','4','5','6','7','8','9','10']

sn.heatmap(cm,fmt='g', cmap='Blues', annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick)
plt.tight_layout()
plt.title('Confusion Matrix of 10 kinds of fonts')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("./confusion_matrix.jpg")
plt.show()