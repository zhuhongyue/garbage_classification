# encoding: utf-8
import numpy as np
import cv2
import time
import flask, json
from flask import Response, request
import base64
import hashlib
import io
import json
import flask
import torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
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


# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
checkpoint_path = "/home/zjfan/zhuhongyue_unionpay/rubbish_classification/checkpoint/best_checkpoint.pth.tar"

data_path = "/home/zjfan/zhuhongyue_unionpay/rubbish_classification/data_rec/"
dataset=ImageFolder(data_path)
class_index = {'ElectricWire': 17, 'EggShell': 16, 'Leaf': 25, 'PlasticCase': 42, 'PaperBag': 35, 'Vegetables': 53, 'ColorPage': 13, 'CircuitBoard': 12, 'Paint': 34, 'Pen': 38, 'Flower': 19, 'Tape': 47, 'Board': 3, 'Medicine': 28, 'OrangePeel': 32, 'Rice': 44, 'Kerne': 24, 'prawn': 57, 'PitayaPeel': 40, 'BananaPeel': 1, 'MangoPeel': 26, 'Outle': 33, 'Newspape': 31, 'TeaResidue': 48, 'Envelope': 18, 'Diapers': 15, 'Glass': 21, 'PearPeel': 37, 'Tissue': 49, 'CantaloupePeel': 9, 'Butt': 8, 'Fluorescent': 20, 'Cardboard': 10, 'fish': 56, 'Bread': 5, 'PlasticFoam': 43, 'Towel': 51, 'Meat': 27, 'GrapefruitPeel': 23, 'ApplePeel': 0, 'Metal': 30, 'Cray': 14, 'StainedPlasticBag': 46, 'Shell': 45, 'PlasticBottle': 41, 'TreeBranch': 52, 'Bulb': 7, 'PineapplePeel': 39, 'Glasses': 22, 'MessTin': 29, 'Battery': 2, 'Ceramic': 11, 'Briefs': 6, 'PeachPeel': 36, 'Book': 4, 'chopsticks': 55, 'ToiletTissue': 50, 'WatermelonPeel': 54}
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model
    model = models.vgg16(pretrained=True)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, len(dataset.classes)))
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    if use_gpu:
        model.cuda()
def prepare_image(image):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """
    if image.mode != 'RGB':
        image = image.convert("RGB")
    img=transform(image)
    img = img.unsqueeze(0)
    img = Variable(img)
    if use_gpu:
        img = img.cuda()
    return torch.autograd.Variable(img, volatile=True)

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    # Ensure an image was properly uploaded to our endpoint.
    base64image = flask.request.values.get('image')  # 获取参数
    if base64image:
        print "base64image is not empty"
    else:
        base64image = request.json['image']
    mgArr = base64image.split(',')
    if(len(imgArr) > 1):
        base64image = imgArr[1]
    else:
        base64image = imgArr[0]
    imageData = base64.b64decode(base64image)
    nparr = np.fromstring(imageData, np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    # Preprocess the image and prepare it for classification.
    img = prepare_image(img)
    # Classify the input image and then initialize the list of predictions to return to the client.
    score = model(img)
    probability = torch.nn.functional.softmax(score,dim=1)
    maxk = max((1,5))
    probly, pred = probability.topk(maxk, 1, True, True)
    pred = pred.cpu().numpy()
    probly = probly.detach().cpu().numpy()
    output = []
    count = 0
    for i,j in zip(probly[0],pred[0]):
        if i > 0.5 :
            count+=1
        print i
        print class_index.keys()[class_index.values().index(j)]
        output.append({class_index.keys()[class_index.values().index(j)]:i})
    if count == 0:
        print('未检测到对象')
        out = {"err_no": "1",
               "err_msg": "未检测到对象"}
    else:
        out = {"ret": output,
               "err_no": "0",
               "err_msg": ""}
    # Return the data dictionary as a JSON response.
    return Response(json.dumps(out, ensure_ascii=False),
                                mimetype='application/json')

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run()

