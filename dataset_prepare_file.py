# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 08:57:51 2021

@author: Abdul Qayyum
"""

#%% dataset prepartion for DFUC2021 challenege
#%compute the class dirtibution
import os
import pandas as pd
import cv2

path="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\train.csv"
impath="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\images"
classpath1="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\Classimages\\Normal"
classpath2="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\Classimages\\infection"
classpath3="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\Classimages\\ischaemia"
classpath4="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\Classimages\\both"
df=pd.read_csv(path)
patricipant=df["image"]
data={"sample":[],
      "label":[]}
for sub_id in patricipant:
    this_pheno = df[df['image'] == sub_id]
    if (this_pheno['none']==1).all():
        data["sample"].append(sub_id)
        imge=cv2.imread(os.path.join(impath,sub_id))
        cv2.imwrite(classpath1+"\\"+str(sub_id)+'_'+'.png',imge)
        data["label"].append(0)
    elif (this_pheno['infection']==1).all():
        data["sample"].append(sub_id)
        data["label"].append(1)
        imge=cv2.imread(os.path.join(impath,sub_id))
        cv2.imwrite(classpath2+"\\"+str(sub_id)+'_'+'.png',imge)
    elif (this_pheno['ischaemia']==1).all():
        data["sample"].append(sub_id)
        imge=cv2.imread(os.path.join(impath,sub_id))
        cv2.imwrite(classpath3+"\\"+str(sub_id)+'_'+'.png',imge)
        data["label"].append(2)
    elif (this_pheno['both']==1).all():
        data["sample"].append(sub_id)
        data["label"].append(3)
        imge=cv2.imread(os.path.join(impath,sub_id))
        cv2.imwrite(classpath4+"\\"+str(sub_id)+'_'+'.png',imge)
        
dffile=pd.DataFrame.from_dict(data)

# cross validation prepartion
import os
import pandas as pd

path="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\train.csv"
df=pd.read_csv(path)
patricipant=df["image"]
data=[]
for sub_id in patricipant:
    this_pheno = df[df['image'] == sub_id]
    if (this_pheno['none']==1).all():
        data.append((sub_id,0))
    elif (this_pheno['infection']==1).all():
        data.append((sub_id,1))
    elif (this_pheno['ischaemia']==1).all():
        data.append((sub_id,2))
    elif (this_pheno['both']==1).all():
        data.append((sub_id,3))
# dictionary to list conversion    
dd=data


# training and validation split
import random
# random.seed(0)
def Trian_val(data_list,test_size=0.15):
    n=len(data_list)
    m=int(n*test_size)
    test_item=random.sample(data_list,m)
    train_item=list(set(data_list)-set(test_item))
    return train_item,test_item
#tr_list,test_list=Trian_val(dd,test_size=0.15)
Fold=5
PATH="D:\\MICCAI2021\\DFU_form\\DFUC2021_trainset_2104271\\DFUC2021_train\\crossvalid"
def crossfold(PATH,Fold,data_list):
    for i in range(Fold):
        tr_list,test_list=Trian_val(data_list,test_size=0.15)
        train_df= pd.DataFrame(tr_list,columns=["sample","label"])
        test_df= pd.DataFrame(test_list,columns=["sample","label"])
        train_df.to_csv(os.path.join(PATH,"Training_"+str(i)+'.csv'), index=False)
        test_df.to_csv(os.path.join(PATH,"Testing_"+str(i)+'.csv'), index=False)

        
crossfold(PATH,Fold,data_list=dd)        

