# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:07:16 2021

@author: Abdul Qayyum
"""
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


#%%
######################################### multimodel ###################################
import timm
print("Available Vision Transformer Models: ")
print(timm.list_models("vit*"))
#################################################### transformer model #############################
import timm
import torch.nn as nn
class classify_layer(nn.Module):
    def __init__(self,in_features,num_classes):
        super(classify_layer,self).__init__()
        self.classifier=nn.Sequential(nn.Linear(in_features,768),
                                      nn.ReLU(True),
                                      nn.Linear(768,num_classes))
        print(self.classifier)
        
    def forward(self,x):
        x=self.classifier(x)
        return x
#################################################### transformer model #############################
class Mutltimodel_DFUC(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(Mutltimodel_DFUC, self).__init__()

        self.model1 = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        #self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        self.model1.head= nn.Sequential(nn.Linear(self.model1.head.in_features,self.model1.head.in_features),
                                                  )
        
        self.model2 = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained)
        #self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        self.model2.head= nn.Sequential(nn.Linear(self.model2.head.in_features,self.model2.head.in_features),
                                                  )
        
        self.fc=nn.Sequential(nn.Linear(3072,768),
                              nn.Dropout(0.3),
                              nn.ReLU(True),
                              )

        self.classify=classify_layer(768,4)


    def forward(self, x):
        F1=self.model1(x)
        F2=self.model2(x)
        # pairwise concatenation
        Concat1=torch.cat((F1,F2),dim=1) # 768+768=512
        Concate2=torch.cat((F2,F1),dim=1) # 768+768=512
        print(Concat1.shape)
        print(Concate2.shape)
        #features = torch.cat((F1, F2), dim=1)  # 768+768=1024
        features = torch.cat((Concat1, Concate2), dim=1)
        #print(features.shape)
        features=self.fc(features)
        score=self.classify(features)
        return score
    
model = Mutltimodel_DFUC(n_classes=4, pretrained=True).to(device)
inp=torch.rand(1,3,224,224).to(device)
out=model(inp)
print(out.shape)
# define loss function and optimzer
#loss_func=nn.CrossEntropyLoss()
import torch 
import torch.nn as nn
classes=['Normal','infection','ischaemia','both']
my_distribution=np.array([2552,2552,227,621])
class_weights = torch.from_numpy(np.divide(1, my_distribution)).float().to(device)
class_weights = class_weights / class_weights.sum()
for i, c in enumerate(classes):
  print('Weight for class %s: %f' % (c, class_weights.cpu().numpy()[i]))
#loss_func = nn.CrossEntropyLoss(weight=class_weights)
loss_func = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer=optim.Adam(model.parameters(),lr=0.0001)
# #%%
#%% training and testing
#%training and testing the deep learning models
# training function and validation
epochs=50
train_loss_ep=[]
train_accuracy_ep=[]
val_loss_ep=[]
val_accuracy_ep=[]
best_acc=0
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(epochs):
    model.train()
    counter=0
    training_run_loss=0.0
    train_running_correct=0.0
    for i, data in tqdm(enumerate(train_loader),total=int(len(dataset_train)/train_loader.batch_size)):
        counter+=1
        # extract dataset
        imge,label=data
        imge=imge.to(device)
        label=label.to(device)
        # zero_out the gradient
        optimizer.zero_grad()
        output=model(imge)
        loss=loss_func(output,label)
        training_run_loss+=loss.item()
        _,preds=torch.max(output.data,1)
        train_running_correct+=(preds==label).sum().item()
        loss.backward()
        optimizer.step()
    train_loss=training_run_loss/len(train_loader.dataset)
    train_loss_ep.append(train_loss)
    train_accuracy=100.* train_running_correct/len(train_loader.dataset)
    train_accuracy_ep.append(train_accuracy)
    print(f"Train Loss:{train_loss:.4f}, Train Acc:{train_accuracy:0.2f}")
    
    # evluation start
    print("validation start")
    
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i,data in tqdm(enumerate(valid_loader),total=int(len(dataset_valid)/valid_loader.batch_size)):
            imge,label=data
            imge=imge.to(device)
            label=label.to(device)
            output=model(imge)
            loss=loss_func(output,label)
            val_running_loss+=loss.item()
            _,pred=torch.max(output.data,1)
            val_running_correct+=(pred==label).sum().item()
        val_loss=val_running_loss/len(valid_loader.dataset)
        val_loss_ep.append(val_loss)
        val_accuracy=100.* val_running_correct/(len(valid_loader.dataset))
        val_accuracy_ep.append(val_accuracy)
        print(f"epoch:{epoch}")
        print(f"Val Loss:{val_loss:0.4f}, Val_Acc:{val_accuracy:0.2f}")
        if val_accuracy>best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
PATH="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\save_weights"
model_name="vit_base_patch16_224_miil_2"
model_path = f"{PATH}/{model_name}_{best_acc}.pth"
torch.save(best_model_wts, model_path)

