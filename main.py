#%%
#external imports
from pickle import FALSE, TRUE
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split    
from torchvision import transforms
from torch.utils.data import DataLoader
import math
from torch import nn
import timm

#internal imports
from dataset import PTB_Dataset
from train import train
#from predict import predict
from validation import validate

#constants
from constants import REC_PATH,CSV_PATH,DATASET_LIMIT,BATCH_SIZE,N_LEADS,N_CLASSES
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#
CustomDataset = PTB_Dataset(CSV_PATH,REC_PATH,transforms.ToTensor())
#print(CustomDataset.__len__())

#Train %
TRAIN_SIZE = math.floor(CustomDataset.__len__()*0.75)

train_dataset,test_dataset = torch.utils.data.random_split(CustomDataset,[TRAIN_SIZE, CustomDataset.__len__()-TRAIN_SIZE])
"""print("train dataset:")
print(train_dataset)
print("test datase:")
print(test_dataset)"""
#X_train,X_test,y_train,y_test = train_test_split(CustomDataset.data[''])

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE)
print(f'test_dataset {test_dataset}')
model = timm.create_model('resnet18',pretrained=True,in_chans=1)
#model=model.to(device)
loss_fn = nn.MSELoss()
print('a')
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#%%
train(train_dataloader=train_dataloader,model=model,loss_fn=loss_fn,optimizer=optimizer)
validate(validate_dataloader=test_dataloader,model=model)


# %%
