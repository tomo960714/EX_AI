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

def train(train_dataloader,model,optimizer,loss_fn,device,size):
    ds_size= train_dataloader.dataset.__len__()
    
    loss_history=np.empty([NEPOCHS,int(np.ceil(size/BATCH_SIZE))],dtype=float)
    output_history = np.empty([NEPOCHS,size])
    true_history = []
    model.train()
    for iEpoch in range(NEPOCHS):
        batch_cnt = 0
        for batch, (input,target) in enumerate(train_dataloader):
            input = input.float()
            target=target.float()
            #print(input.shape)
            input=input.to(device)
            target=target.to(device)
            
            #zero the gradient
            optimizer.zero_grad()
            
            #perform forward pass
            output=model(input)
            output_1=torch.movedim(output,1,0)
            output_new=nn.functional.softmax(output_1[0],dim=0)
        
            
            #calcualting loss
            loss = loss_fn(output_new,target)
            """ run['train/epochs/loss'].log(loss)
            run['train/epochs/output'].log(output_new.detach().cpu().numpy())
            run['train/epochs/target'].log(target.detach().cpu().numpy())"""
            tmp_out=output_new.detach().cpu().numpy()
            tmp_true =target.detach().cpu().numpy()
            for i in range(len(tmp_out)):
                if iEpoch==0:
                    true_history.append(tmp_true[i,])
                    
                output_history[iEpoch,i+(BATCH_SIZE*batch_cnt)]=tmp_out[i,]
        

            loss_history[iEpoch,batch_cnt]=loss.item()
            batch_cnt += 1
            #print(loss.item())
           
            #perform backward pass
            loss.backward()
            
            #perform optimization
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(input)
                print(f"loss: {loss:>7f}  [{current:>5d}/{ds_size:>5d}] Epoch [{iEpoch+1:>3d}/{NEPOCHS:>3d}] ")
            
    #TODO: To save loss, should I onyl use loss that is up there or all of the loss values?
    #print("loss history:",loss_history)    
    print("out_history:",output_history)
    return output_history,true_history,loss_history