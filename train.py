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
import time
import copy
from GPUtil import showUtilization as gpu_usage
import csv
import os
import tqdm
#internal imports
from dataset import PTB_Dataset

#constants
from constants import REC_PATH,CSV_PATH,DATASET_LIMIT,BATCH_SIZE,N_LEADS,N_CLASSES,NEPOCHS
from utils import tensor2list as t2l

def train_loop(dataloaders,model,optimizer,loss_fn,device,weights_name,evaluate = True):

    if evaluate == True:
        phases =['Train', 'Valid']
    else:
        phases = ['Train']
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    fieldnames_metrics = ['epoch', 'Train_loss', 'Valid_loss']
    fieldnames_values = ['epoch','Train_target', 'Train_pred','Valid_target','Valid_pred']
    with open('results/metrics.csv', 'w', newline='') as csvfile:
        metric_writer = csv.DictWriter(csvfile, fieldnames=fieldnames_metrics)
        metric_writer.writeheader()
    with open('results/values.csv', 'w', newline='') as csvfile:
        value_writer = csv.DictWriter(csvfile, fieldnames=fieldnames_values)
        value_writer.writeheader()

    for epoch in range(1,NEPOCHS+1):
        gpu_usage()
        #gpu_usage()
        print('Epoch {}/{}'.format(epoch, NEPOCHS))
        print('-' * 10)
        # Each epoch has a training and Validation phase
        # Initialize batch summary
        metrics_summary = {a: [0] for a in fieldnames_metrics}
        
        value_summary = {a: [0] for a in fieldnames_values}
        print(metrics_summary)

        for phase in phases:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for (inputs,target) in iter(dataloaders[phase]):
                inputs = torch.squeeze(inputs, dim=1)
                inputs = inputs.float().to(device)
                target=target.float().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, target)

                    try:
                        if phase == 'Train':
                            metrics_summary['Train_loss'].append(loss.item())
                            value_summary['Train_target'].append(t2l(target))
                            value_summary['Train_pred'].append(t2l(outputs))
                        else:
                            metrics_summary['Valid_loss'].append(loss.item())
                            value_summary['Valid_target'].append(t2l(target))
                            value_summary['Valid_pred'].append(t2l(outputs))
                    except:
                        metrics_summary[f'{phase}_loss'].append(0)
                        value_summary[f'{phase}_target'].append(0)
                        value_summary[f'{phase}_pred'].append(0)
                # backward + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()
            inputs = None
            target = None
            metrics_summary['epoch'] = epoch
            value_summary['epoch'] = epoch
            print(value_summary['Train_target'])
            print(f'{phase} phase, loss: {loss.item()}')
        
        with open('results/metrics.csv', 'a', newline='') as csvfile:
            metric_writer = csv.DictWriter(csvfile, fieldnames=fieldnames_metrics)
            metric_writer.writerow(metrics_summary)
            if phase == 'Valid' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join('results/',weights_name))
        with open('results/values.csv', 'a', newline='') as csvfile:
            value_writer = csv.DictWriter(csvfile, fieldnames=fieldnames_values)
            value_writer.writerow(value_summary)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    
    return model

"""
            for batch, (input,target) in enumerate(data_loaders[phase]):
                input = torch.squeeze(input, dim=1)
                input = input.float()
                target=target.float()
                #print(input.shape)
                input=input
                target=target
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
                """
            

    #TOO: To save loss, should I onyl use loss that is up there or all of the loss values?
    #prnt("loss history:",loss_history)    
    #prnt("out_history:",output_history)
    #return pred_history,true_history,loss_tmp/nr_entries