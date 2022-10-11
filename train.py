# %%
#external imports
from pickle import FALSE, TRUE
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split    
from torchvision import transforms

from torch.utils.data import DataLoader
import math
import torchvision

from transformers import ResNetConfig, ResNetModel
#internal imports
from dataset import PTB_Dataset
from resnet import resnet18

#constants
from constants import REC_PATH,CSV_PATH,DATASET_LIMIT,BATCH_SIZE,N_LEADS,N_CLASSES,NEPOCHS


def train(train_dataloader,model,optimizer,loss_fn):

    #preparation

    # Set device
    """device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #
    CustomDataset = PTB_Dataset(CSV_PATH,REC_PATH,transforms.ToTensor())
    #print(CustomDataset.__len__())

    #Train %
    TRAIN_SIZE = math.floor(CustomDataset.__len__()*0.75)

    train_dataset,test_dataset = torch.utils.data.random_split(CustomDataset,[TRAIN_SIZE, CustomDataset.__len__()-TRAIN_SIZE])
    print("train dataset:")
    print(train_dataset)
    print("test datase:")
    print(test_dataset)
    #X_train,X_test,y_train,y_test = train_test_split(CustomDataset.data[''])

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE)

    """
    # %%
    ##ehhhhhhh
    #config = ResNetConfig(num_channels=12)
    #, num_classes=12
    #model = timm.create_model('resnet18',pretrained=True,in_chans=1)
    #print(model(torch.rand(1,12,224,224)).shape)
    #print(train_dataloader)

    #loss_fn = nn.CrossEntropyLoss()

    size= train_dataloader.dataset.__len__()
    #print(size)
    """learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)"""


    #%%
    loss_history = []
    batch_history = []
    for iEpoch in range(NEPOCHS):
        for batch,(X,y) in enumerate(train_dataloader):
            """print('X',X)
            print('shape',X.shape)
            print('y',y)
            print('batch',batch)"""
            X=X.float()
            y=y.long()
            
            pred = model(X)
            loss = loss_fn(pred,y)
            #save loss
            #print(loss_history.type)
            
            loss_history.append(loss.detach().cpu().numpy())
            batch_history.append(batch)
            print(batch)
            #print('pred',pred)
            #print('pred shape:',pred.shape)
            """print(cnt)
            cnt = cnt+1"""
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] Epoch [{iEpoch:>3d}/{NEPOCHS:>3d}] ")
    print(loss_history)
    print(batch_history)    



# %%
