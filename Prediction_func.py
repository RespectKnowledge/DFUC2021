# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:09:53 2021

@author: Abdul Qayyum
"""

#%% load the model and check validation accuracy
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

cudnn.benchmark = True

import os 
import pandas as pd
from PIL import Image
import numpy as np
import cv2
# path="/content/drive/MyDrive/DFUC2021_train/images"
# pathcsv="/content/drive/MyDrive/DFUC2021_train/Training_list1.csv"
# readptah=pd.read_csv(pathcsv)
# sample=readptah.values.tolist()[0]
# img,label=sample
# img=np.array(Image.open(os.path.join(path,img)))

# import matplotlib.pyplot as plt
# img.shape
#imge1=img[:,:,1]
#plt.imshow(imge1)

#!pip install albumentations

#! pip install albumentations==0.4.6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
PATH="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\save_weights"
pathload=os.path.join(PATH,"model_vit.pth")
model.load_state_dict(torch.load(pathload))
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valid_loader:
        images, labels = data["sample"],data["label"]
        images=images.float().to(device)
        labels=labels.to(device)
        model=model.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#%
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,512),nn.Dropout(0.5),
#                             nn.ReLU(True),
#                             nn.Linear(512,4),
#                             )
# model= model_ft    
# model.to(device)
#%% prediciton function
########################### prediciton and submission function ###################
root_path="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\Val2021"
lstdir=os.listdir(root_path)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
model.to(device)
pred=[]
data = {
    'image': [],
    'none': [],
    'infection': [],
    'ischaemia': [],
    'both': []
}
import natsort
for i in natsort.natsorted(lstdir):
  path=os.path.join(root_path,i)
  data["image"].append(i)
  image=np.array(Image.open(path))
  image=val_transforms(image=image)["image"]
  #print(image.shape)
  image=image.unsqueeze(0).to(device)
  y_pred=model(image)
  y_ = torch.softmax(y_pred, dim=1)
  print(y_ .shape)
  uu=y_.detach().cpu().numpy()
  pred=np.squeeze(uu, axis=0)
  none,infection,ischaemia,both=pred
  data["none"].append(none)
  data["infection"].append(infection)
  data["ischaemia"].append(ischaemia)
  data["both"].append(both)
  
import pandas as pd
df=pd.DataFrame.from_dict(data)
df.to_csv("predcition_vit_base_patch16_224.csv",index=False)