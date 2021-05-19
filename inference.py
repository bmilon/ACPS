# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:10:31 2021

@author: mbhattac
"""


import torchvision
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
sns.set()

class_path = r"C:\Users\mbhattac\Downloads\results\train\\"

transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.ImageFolder(
    root=class_path ,
    transform=transforms)

# dataloader
base_loader = torch.utils.data.DataLoader(dataset, batch_size=16,shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = "acps.pt"
model = torch.load(PATH)

#model = models.resnet18(pretrained=False)
min_running_val_acc = 0.74

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# split our dataset

train_size=int(len(dataset)*0.8)
test_size=len(dataset) -train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16,shuffle=True, num_workers=0)
        
# create arrays to save metrics
t_start = time.time()
loss_arr=[]
acc_arr=[]
val_acc_arr=[]
epochs=20
for epoch in range(epochs):  # loop over the dataset multiple times
    
    #initialize epoch metrics
    t_epoch = time.time()
    running_loss = 0.0
    running_acc = 0.0
    running_n = 0
    running_val_acc = 0.0
    running_val_n = 0
    
    # train
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        
        # compute minibatch metrics
        running_acc += (torch.argmax(outputs, dim=1).to(device) == labels.to(device)).float().sum()
        running_n += len(labels)
        running_loss += loss.item()
    
    # validate/evaluate
    for i, data in enumerate(val_loader, 0):
        # get the inputs
        inputs, labels = data

        # forward
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        
        # compute minibatch metrics
        running_val_acc += (torch.argmax(outputs, dim=1).to(device) == labels.to(device)).float().sum()
        running_val_n += len(labels)
        
    # compute epoch metrics
    running_acc = running_acc/running_n
    running_val_acc = running_val_acc/running_val_n
    running_loss = running_loss/running_n
    loss_arr.append(running_loss)
    acc_arr.append(running_acc)
    val_acc_arr.append(running_val_acc)
    
    # print epoch metrics
    print('epoch: {} loss: {} acc: {} valacc: {} time epoch: {} time total: {}'.format(epoch + 1, running_loss, running_acc, running_val_acc, time.time()-t_epoch, time.time()-t_start))


    if running_val_acc > min_running_val_acc:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_running_val_acc,running_val_acc))
        # save checkpoint as best model
        torch.save(model, PATH)
        min_running_val_acc = running_val_acc    

print('Finished Training in {} seconds'.format(time.time()-t_start))


val_acc_arr = [a.cpu().detach().numpy().tolist() for a in val_acc_arr]
acc_arr = [a.cpu().detach().numpy().tolist() for a in acc_arr]

sns.set_palette("tab10")
sns.set_context("talk")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(y=val_acc_arr, x=range(epochs),label="Validation Accuracy",legend='brief',markers=True)
ax = sns.lineplot(y=acc_arr,x=range(epochs), label="Training Accuracy",legend='brief',markers=True)
ax.set(ylim=(0, 1),xlim=(0, 25))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("variation  of accuracy with increasing epochs")

plt.legend()
plt.show()        


def matplotlib_imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
        
        
dataiter = iter(val_loader)
images, labels = dataiter.next()
labels_map = ['Healthy','Rust in Leaf','Rust in stem']
list(dataset.class_to_idx.keys())

prediction = model(images.to(device))
prediction=prediction.cpu().detach().numpy().argmax(axis=1)

prediction = [labels_map[a] for a in prediction]
actual_labels = [labels_map[a] for a in labels]

for index, (image,p,a) in enumerate(zip(images,prediction,actual_labels)):
    pass
    ax  = plt.subplot(2,int(images.shape[0]/2),index+1)
    npimg = image.numpy() 
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(f'Actual : {a} \n Predicted : {p}', fontsize = 10)
    plt.axis('off')

plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

#plt.plot(val_acc_arr, label="Validation Accuracy")
#plt.plot(acc_arr, label="Training Accuracy")
#plt.legend()
#plt.show()
#plt.plot(loss_arr, label="Training loss")

#torch.save(model, PATH)