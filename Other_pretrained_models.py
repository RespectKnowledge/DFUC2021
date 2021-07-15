# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:28:58 2021

@author: Abdul Qayyum
"""

#%% other pretrained models that are used for testing the DFUC2021 dataset 
# and not used in this challenege
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
#https://pytorch.org/get-started/previous-versions/
cudnn.benchmark = True
import torch
print(torch.__version__)
#dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class DataSetDFUCTrain(Dataset):
    def __init__(self,root,csv_path,tranform):
        super().__init__()
        self.root=root
        self.csv_path=csv_path
        self.tranform=tranform
        self.readfile=pd.read_csv(self.csv_path)
        
    def __getitem__(self,idx):
        sample=self.readfile.values.tolist()[idx]
        img,label=sample
        imagepath=os.path.join(self.root,img)
        image=np.array(Image.open(imagepath))
        if self.tranform is not None:
            image=self.tranform(image=image)["image"]
        return {"sample":image,
                     "label":label}
    def __len__(self):
        return len(self.readfile)
    
device= "cuda" if torch.cuda.is_available() else "cpu"  
print(device) 
# Data augmentation for images
train_transforms = A.Compose([A.Resize(width=224, height=224),
                              A.RandomCrop(height=224, width=224),
                              A.HorizontalFlip(p=0.5),
                              A.VerticalFlip(p=0.5),
                              A.RandomRotate90(p=0.5),
                              A.Blur(p=0.3),
                              A.CLAHE(p=0.3),
                              #A.ColorJitter(p=0.3),
                              A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
                              A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
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

import os 
import pandas as pd
from PIL import Image
import numpy as np
import cv2
device = "cuda" if torch.cuda.is_available() else "cpu"
path="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\images"
pathcsv="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\Training_list1.csv"
pathvalid="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\valid_list1.csv"
dataset_train=DataSetDFUCTrain(path,pathcsv,tranform=train_transforms)
dataset_valid=DataSetDFUCTrain(path,pathvalid,tranform=val_transforms)   

from torch.utils.data import DataLoader
train_loader=DataLoader(dataset_train,batch_size=20,pin_memory=True,
                        shuffle=True)
valid_loader=DataLoader(dataset_valid,batch_size=20,pin_memory=True,
                        shuffle=False)
#%
# # pytorch petrained models for classification
def premodels(pretrained,model_selec,num_classes):
    
    if model_selec=="ResNet":
        model_ft = models.resnet18(pretrained=pretrained)
        ## Modify fc layers to match num_classes
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
    
    elif model_selec=="DensNet":
        model = models.densenet201(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier= nn.Linear(num_ftrs, num_classes)
        
    elif model_selec=="SENet":
        model_ft = models.squeezenet1_0(pretrained=pretrained)
        for params in list(model_ft.parameters())[0:-5]:
            params.requires_grad = False
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model=model_ft
    
    elif model_selec=="MobileNet":
        model_ft = models.mobilenet_v2(pretrained=pretrained)    
        # Freeze all the required layers (i.e except last conv block and fc layers)
        for params in list(model_ft.parameters())[0:-5]:
            params.requires_grad = False
        # Modify fc layers to match num_classes
        num_ftrs=model_ft.classifier[-1].in_features
        model_ft.classifier=nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True))
        model=model_ft
    
    
    elif model_selec=="VGG11":
        model_ft = models.vgg11_bn(pretrained=pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model=model_ft

    elif model_selec=="Efficents":
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained("efficientnet-b7")
        model._fc = nn.Sequential(nn.Linear(2560, 256),
                                  nn.Dropout(0.5),
                                  nn.ReLU(True),
                                  nn.Linear(256,num_classes),
                                  )
        
    elif model_selec=="Efficents2m":
        model=effnetv2_m()
        
    elif model_selec=="Efficents2s":
        model=effnetv2_s()
        
    elif model_selec=="Efficents2L":
        model=effnetv2_l()
        
    elif model_selec=="Efficents2XL":
        model=effnetv2_xl()
        
    # elif model_selec=="Efficents2m":
    #     model=effnetv2_m()

    else:
        print("no model available")
        
    return model

# model_ft = models.resnet18(pretrained=True)
# ## Modify fc layers to match num_classes
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,512),nn.Dropout(0.5),
#                            nn.ReLU(True),
#                            nn.Linear(512,4),
#                            )
model= premodels(pretrained,model_selec,num_classes) 
model.to(device)

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