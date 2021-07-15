# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:00:54 2021

@author: Abdul Qayyum
"""

#%% Dataset loader file
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

class DataSetDFUCTrain1(Dataset):
    def __init__(self,root,transform):
        super().__init__()
        self.root=root
        self.transform=transform
        normalpathn=os.path.join(self.root,"Normal")
        pathlstn=os.listdir(normalpathn)
        normalpathin=os.path.join(self.root,"infection")
        pathlstin=os.listdir(normalpathin)
        normalpathis=os.path.join(self.root,"ischaemia")
        pathlstis=os.listdir(normalpathis)
        normalpathb=os.path.join(self.root,"both")
        pathlstb=os.listdir(normalpathb)
        
        self.classn=[]
        for cl1 in pathlstn:
            pathn=os.path.join("Normal",cl1)
            self.classn.append((pathn,0))
        self.classin=[]
        for cl2 in pathlstin:
            pathin=os.path.join("infection",cl2)
            self.classin.append((pathin,1))
    
        self.classis=[]
        for cl3 in pathlstis:
            pathis=os.path.join("ischaemia",cl3)
            self.classis.append((pathis,2))
    
        self.classb=[]
        for cl4 in pathlstb:
            pathb=os.path.join("both",cl4)
            self.classb.append((pathb,3))
    
        self.fulllist=self.classn+self.classin+self.classis+self.classb
        #print(self.fulllist)
        
    def __getitem__(self,idx):
        
        paths,label=self.fulllist[idx]
        #print(sample)
        imagepath=os.path.join(self.root,paths)
        #print(imagepath)
        image=np.array(Image.open(imagepath))
        #print(image.shape)
        
        if self.transform is not None:
            image=self.transform(image=image)["image"]
        image=image
        label=label
        
        return image,label
    
    def __len__(self):
        return len(self.fulllist)


#pathtrain="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\newdataset\\Classimages1-20210714T084847Z-001\\dfucdataset\\train"
#pathval="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\newdataset\\Classimages1-20210714T084847Z-001\\dfucdataset\\val"

# Data augmentation for images
train_transforms = A.Compose([A.Resize(width=224, height=224),
                              A.RandomCrop(height=224, width=224),
                              # A.HorizontalFlip(p=0.5),
                              # A.VerticalFlip(p=0.5),
                              # A.RandomRotate90(p=0.5),
                              # A.Blur(p=0.3),
                              # A.CLAHE(p=0.3),
                              # #A.ColorJitter(p=0.3),
                              # A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
                              # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
                              A.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5],
                                          ),
                              ToTensorV2(),
                              ])

val_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                ),
    ToTensorV2(),
    ]
    )


pathtrain="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\newdataset\\Classimages1-20210714T084847Z-001\\dfucdataset\\train"
pathval="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\newdataset\\Classimages1-20210714T084847Z-001\\dfucdataset\\val"

dataset_train=DataSetDFUCTrain1(pathtrain,transform=train_transforms)
dataset_valid=DataSetDFUCTrain1(pathval,transform=val_transforms)   
# len(dataset_train)
# for i in range(len(dataset_train)):
#     print(i)
# image,label=dataset_train[0]

from torch.utils.data import DataLoader
train_loader=DataLoader(dataset_train,batch_size=20,pin_memory=True,
                        shuffle=True)
valid_loader=DataLoader(dataset_valid,batch_size=20,pin_memory=True,
                        shuffle=False)
# images,labels=next(iter(train_loader))
# print(images.shape)
# print(labels)