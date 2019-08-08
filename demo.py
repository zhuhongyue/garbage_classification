from PIL import Image
import torchvision.transforms
from torch.autograd import Variable 
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
import pdb
use_gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
checkpoint_path = "./checkpoint/best_checkpoint.pth.tar"
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
data_path = "./data_rec/"
dataset=ImageFolder(data_path)
class_index = dataset.class_to_idx
img = Image.open('./demo.jfif')  
img=transform(img)
img = img.unsqueeze(0)
img = Variable(img)
if use_gpu:
    img = img.cuda()
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
model.eval()
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
print("the best accary in checkpoint is {}".format(torch.load(checkpoint_path)['accuary']))
score = model(img)
probability = torch.nn.functional.softmax(score,dim=1)
maxk = max((1,5))
probly, pred = probability.topk(maxk, 1, True, True)
pred = pred.cpu().numpy()
probly = probly.detach().cpu().numpy()
for i,j in zip(probly[0],pred[0]):
    print i
    print class_index.keys()[class_index.values().index(j)]

