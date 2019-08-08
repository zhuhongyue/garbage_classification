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
from tqdm import tqdm
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
validation_split = .2
random_seed= 42
batch_size = 16
use_gpu = 1
data_path = "./data_rec/"
checkpoint_path = "./checkpoint/best_checkpoint.pth.tar"
np.random.seed(random_seed)
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
dataset=ImageFolder(data_path,transform = transform)
train_size = int((1-validation_split) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers=1)
model = models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, len(dataset.classes)))

model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
print("the best accary in checkpoint is {}".format(torch.load(checkpoint_path)['accuary']))
import pdb
pdb.set_trace()
print("testing model in test dataset")
if use_gpu:
    model = model.cuda()
acc = 0.0
for i,data in enumerate(tqdm(testloader),0):
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model(inputs)
    acc += accuracy_score(labels.cpu().numpy(), outputs.argmax(1).cpu().numpy())
print("accuracy in test dataset is  {}".format(acc/len(testloader)))

