import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from utils import load_data
from models import *
import numpy as np


import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
# input_size = 16384 #128x128
# hidden_size = 100
# num_classses = 4

num_epochs = 30
batch_size = 64
learning_rate = 0.001
pixel_size = 128

print('--> data load...')
train_dataset, test_dataset, train_data_label = load_data(pixel_size)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


Mode_Index = {"walk": 0, "bike": 1,  "car": 2, "taxi": 2, "bus": 3, "subway": 3, "railway": 3, "train": 3}
classes = ('walk', 'bike', 'car', 'bus')

class_dict={}
for y in train_data_label:
    if y not in class_dict:
        class_dict[y]=1
    else:
        class_dict[y]+=1
class_dict = sorted(class_dict.items(), key=lambda item:item[0])
class_dict = dict(class_dict)
print('All geolife class:', class_dict, class_dict.values())

print('--> Build model...')

# print('ConvNet')
# model = ConvNet().to(device)  ## CNN
print('ResNet18')
model = ResNet18().to(device)   # resnet18

# loss and optimize
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

# training loop
loss_list = []
n_total_steps = len(train_loader)
train_acc = []
test_acc = []


print('--> Model training...', 'epoch:', num_epochs)


best_acc = 0
best_epoch = 0

model.train()
weights = torch.Tensor(list(class_dict.values()))
beta = 0.15
weights = weights.max() / weights
weights = weights/ weights.sum() + beta
weights *= 1/weights.mean() # mean to 1
weights = torch.Tensor(weights).to(device)

print(f'parameters. pixel size: {pixel_size}, num epoches: {num_epochs}, batch_size: {batch_size}, learning rate: {learning_rate}, weight beta, {beta}')
train_loss = []
for epoch in range(num_epochs):

    n_correct = 0
    n_samples = 0
    losses = []

    # training phase
    for i, (images, labels) in enumerate(train_loader):

        # images = images.reshape(-1, 128*128).to(device)
        images = images.to(device)
        labels = labels.to(device)
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        # loss = weights[labels] * loss
        loss = loss.mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        losses.append(loss.item())
    # scheduler.step()
    acc_train = 100.0 * n_correct / n_samples
    print(f'epoch {epoch+1} / {num_epochs}, loss = {np.mean(losses):.4f}, acc = {acc_train:.4f}')  
    train_loss.append(np.mean(losses))
    train_acc.append(acc_train)

    # validation phase
    n_correct = 0
    n_samples = 0

    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]

    test_results = []
    label_list = []


    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        label_list.append(labels.tolist())

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        test_results.append(predictions.tolist())
        # print(predictions.tolist())
        # print(labels.tolist())
        predictions = predictions.tolist()
        labels = labels.tolist()
        # print(len(labels), labels)
      
        for i in range(len(labels)):
            # print(i, labels[i])
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc_val = 100.0 * n_correct / n_samples
    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accurancy of {classes[i]}: {acc:.4f} %')
    print(f'test acc = {acc_val:.4f}')
    test_acc.append(acc_val)

    if acc_val > best_acc:
        best_epoch=epoch+1
        best_acc = max(acc_val, best_acc)

print("best_acc = {:3.1f}({:d})".format(best_acc, best_epoch))

print('loss list:', train_loss)
print('train acc:', train_acc)
print('test acc:', test_acc)
    
# test
print('--> model test...')
model.eval()
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]

    test_results = []
    label_list = []
    losses = []

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        label_list.append(labels.tolist())

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        test_results.append(predictions.tolist())
        # print(predictions.tolist())
        # print(labels.tolist())
        predictions = predictions.tolist()
        labels = labels.tolist()
        
        # for i in range(batch_size):
        #     label = labels[i]
        #     pred = predictions[i]
        #     if (label == pred):
        #         n_class_correct[label] += 1
        #     n_class_samples[label] += 1


    acc = 100.0 * n_correct / n_samples
    # print(f'accuracy = {acc:.4f}')

    # for i in range(4):
    #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #     print(f'Accurancy of {classes[i]}: {acc} %')



# print(test_results[:5])
# print(label_list[:5])



