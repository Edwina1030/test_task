import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataloader import TorchDataset
from model import build_model
import copy

train_val_filename = "./train_val.txt"
train_val_dir = "./target_data/train_val"
train_val_set = TorchDataset(filename=train_val_filename, image_dir=train_val_dir, repeat=1)
train_set, val_set = torch.utils.data.random_split(train_val_set, [1150, 287])

print('number of images in training dataset:', len(train_set))
print('number of images in validation dataset:', len(val_set))

model = build_model()
batch_size = 64 # relative better
# 128 better performance at training and validation data sets, worse at testing dataset
# 32 worse performance at all data sets
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# comparing with optimizer SGD,ASGD etc, Adam peformances still better
# using dichotomy, within 0.1 and 0.0001, get relative better performance with lr 0.001
nums_epoch = 15
# tranining 10-30 epoch, 15 is enough
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

print('====== start training and validation =====')

train_loss_logs = []
train_acc_logs = []
val_loss_logs = []
val_acc_logs = []

for epoch in range(nums_epoch):

    train_loss = 0
    train_acc = 0
    model = model.train()
    for img, label in train_loader:
        img = Variable(img)
        label = Variable(label)

        out = model(img)
        loss = criterion(out, label.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label.reshape(-1)).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    train_loss_logs.append(train_loss / len(train_loader))
    train_acc_logs.append(train_acc / len(train_loader))

    val_loss = 0
    val_acc = 0
    for img, label in val_loader:
        img = Variable(img)
        label = Variable(label)

        out = model(img)
        loss = criterion(out, label.reshape(-1))
        val_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label.reshape(-1)).sum().item()
        acc = num_correct / img.shape[0]
        val_acc += acc

    val_loss_logs.append(val_loss / len(val_loader))
    val_acc_logs.append(val_acc / len(val_loader))

    print('Epoch', epoch + 1)
    print('Training Loss:', train_loss / len(train_loader), 'Training Accuracy:', train_acc / len(train_loader))
    print('Validation Loss:', val_loss / len(val_loader), 'Validation Accuracy:', val_acc / len(val_loader))

min_val_loss = 1
best_model = None
min_epoch = 10
for epoch in range(nums_epoch):
    if epoch > min_epoch and val_loss <= min_val_loss:
        min_loss_val = val_loss
        best_model = copy.deepcopy(model)
model = best_model
torch.save(model, './model.pkl')
print('===='*10)
print('best model is saved')

x = range(nums_epoch)
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(x, train_loss_logs, "r", marker='o', ms=5, label="Training Loss")
plt.plot(x, val_loss_logs, "g", marker='o', ms=5, label="Validation loss")
plt.xticks(rotation=45)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc="upper left")

plt.subplot(2, 1, 2)
plt.plot(x, train_acc_logs, "r", marker='o', ms=5, label="Training Acc")
plt.plot(x, val_acc_logs, "g", marker='o', ms=5, label="Validation acc")
plt.xticks(rotation=45)
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.title("Training and Validation Acc")
plt.legend(loc="upper left")

plt.savefig("./learning_curve.jpg",)
plt.show()
