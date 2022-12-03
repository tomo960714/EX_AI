import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split    
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import math
import torchvision
import neptune.new as neptune
#internal imports
from dataset import PTB_Dataset

#constants
from constants import REC_PATH,CSV_PATH,DATASET_LIMIT,BATCH_SIZE,N_LEADS,N_CLASSES,NEPOCHS
from utils import tensor2list as t2l

def train_loop(train_dataloader,model,optimizer,loss_fn,device,size):
    pred_history = []
    true_history = []
    loss_tmp = 0
    nr_entries = 0
    model.train()
    
    for batch, (input,target) in enumerate(train_dataloader):
        input = torch.squeeze(input, dim=1)
        input = input.float()
        target=target.float()
        #print(input.shape)
        input=input.to(device)
        target=target.to(device)
        #print(input.shape)
        #%%
        #perform forward pass
        output=model(input)
        #print(output.shape)
        #calcualting loss
        loss = loss_fn(output,target)
        #zero the gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_history = pred_history + output.detach().cpu().numpy().tolist()
        true_history = true_history + target.detach().cpu().numpy().tolist()
        loss_tmp += loss.detach().cpu().numpy()
        nr_entries += len(input)
        #print(loss_tmp)
    print(f"train loss:{loss_tmp}}")

    #TOO: To save loss, should I onyl use loss that is up there or all of the loss values?
    #prnt("loss history:",loss_history)    
    #prnt("out_history:",output_history)
    return pred_history,true_history,loss_tmp/nr_entries