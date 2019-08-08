from torchvision import models
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time
from sklearn.metrics import accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
#load checkpoint function
def load_checkpoint(checkpoint_model, checkpoint_PATH, checkpoint_optimizer):
    import pdb
    model_CKPT = torch.load(checkpoint_PATH)
    pdb.set_trace()
    checkpoint_model.load_state_dict(model_CKPT['state_dict'])
    checkpoint_optimizer.load_state_dict(model_CKPT['optimizer'])
    checkpoint_acc = model_CKPT['accuary']
    return checkpoint_model, checkpoint_optimizer,checkpoint_acc

#param
num_epochs = 20
validation_split = .2
random_seed= 42
batch_size = 64
use_gpu = 1
use_checkpoint = 1
data_path = "/home/zjfan/zhuhongyue_unionpay/rubbish_classification/data_rec/"
checkpoint_path = "/home/zjfan/zhuhongyue_unionpay/rubbish_classification/checkpoint/best_checkpoint.pth.tar"

#dataset prepare
np.random.seed(random_seed)
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
dataset=ImageFolder(data_path,transform = transform)
import pdb
pdb.set_trace()
train_size = int((1-validation_split) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers=1)

#model
model = models.vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad = False
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, len(dataset.classes)))
if use_gpu:
    model = model.cuda()    
optimizer = torch.optim.Adam(model.classifier.parameters(),weight_decay = 0.05)
criterion = nn.CrossEntropyLoss()
if use_checkpoint:
    load_checkpoint(model,checkpoint_path,optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5, eta_min=4e-08)

#train
print('Start Traning')
from tqdm import tqdm
best_acc = 0.0
for epoch in range(num_epochs):  # loop over the dataset multiple times
    scheduler.step()
    running_loss = 0.0
    model.train()
    print("training {}/{}:".format(epoch,num_epochs))
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 5 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    model.eval()
    acc = 0.0
    print("testing {}/{}:".format(epoch,num_epochs))
    for i,data in enumerate(tqdm(testloader),0):
    	inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
    	outputs = model(inputs)
    	acc += accuracy_score(labels.cpu().numpy(), outputs.argmax(1).cpu().numpy())
    acc = acc/len(testloader)
    print("accuracy in {}th epoch is  {}".format(epoch,acc))
    if acc > best_acc:
        best_acc = acc
        torch.save({'accuary': best_acc, 
                    'class_to_idx':dataset.class_to_idx,
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict()
                    },
                    "./checkpoint/best_checkpoint.pth.tar")
print('Finished Training')

#test
# model = model.load_state_dict(torch.load(PATH))
# for i,date in enumerate(testloader,0):
# 	inputs, labels = data
# 	outputs = model(inputs)