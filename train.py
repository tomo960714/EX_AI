import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split    
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import math
import torchvision
#import neptune.new as neptune
import time
import copy
from GPUtil import showUtilization as gpu_usage
import csv
import os
import tqdm
import shutil
#internal imports
from dataset import PTB_Dataset


#constants
from constants import REC_PATH,CSV_PATH,N_LEADS,N_CLASSES
from utils import tensor2list as t2l

<<<<<<< HEAD
def train_loop(dataloaders,model,optimizer,loss_fn,device,weights_name,epochs,set_sizes,evaluate = True):
    
=======
def train_loop(dataloaders,model,optimizer,loss_fn,device,weights_name,epochs,evaluate = True):

>>>>>>> 859bb1da9020ce72d3e50248cefc3e7f8c7034d8
    if evaluate == True:
        phases =['Train', 'Valid']
    else:
        phases = ['Train']
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    fieldnames_metrics = ['epoch', 'Train_loss', 'Valid_loss']
    fieldnames_values = ['epoch','Train_target', 'Train_pred','Valid_target','Valid_pred']
    df_metrics = pd.DataFrame(columns=fieldnames_metrics)
    df_values = pd.DataFrame(columns=fieldnames_values)
    save_path=os.path.join('results/',weights_name)

    for epoch in range(1,epochs+1):
<<<<<<< HEAD
        epoch_loss = 0.0
        best_model_changed = False
=======
        gpu_usage()
>>>>>>> 859bb1da9020ce72d3e50248cefc3e7f8c7034d8
        #gpu_usage()
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        # Each epoch has a training and Validation phase
        # Initialize batch summary
        metrics_summary = {'Train_loss': [], 'Valid_loss': []}
        #print(metrics_summary)
        train_targets = []
        train_predictions = []
        valid_targets = []
        valid_predictions = []

       
        for phase in phases:
            running_loss = 0.0
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
                    running_loss += loss.item() * inputs.size(0)

                    try:
                        if phase == 'Train':

                            #metrics_summary['Train_loss']=loss.detach().cpu().numpy()
                            
                            train_predictions = train_predictions + outputs.detach().cpu().numpy().tolist()
                            train_targets = train_targets + target.detach().cpu().numpy().tolist()
                            

                        else:
                            #metrics_summary['Valid_loss']=loss.item()
                            valid_predictions = valid_predictions + outputs.detach().cpu().numpy().tolist()
                            valid_targets = valid_targets + target.detach().cpu().numpy().tolist()
                    except:
                        if phase == 'Train':
                            train_predictions = 0
                            train_targets = 0
                        else:
                            valid_predictions = 0
                            valid_targets = 0
                       # metrics_summary[f'{phase}_loss']=0
                        #value_summary[f'{phase}_target'].append(0)
                        #value_summary[f'{phase}_pred'].append(0)
                # backward + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

            inputs = None
            target = None

            print(f'{phase} phase, loss: {loss.item()}')
            try:
                if phase == 'Train':
                    metrics_summary['Train_loss']=running_loss.detach().cpu().numpy()
                else:    
                    metrics_summary['Valid_loss']=running_loss.detach().cpu().numpy()
            except:
                    metrics_summary[f'{phase}_loss']=0 

        epoch_loss = running_loss / set_sizes[phase]
       
        df_values = df_values.append({'epoch':epoch,'Train_target':train_targets,'Train_pred':train_predictions,'Valid_target':valid_targets,'Valid_pred':valid_predictions},ignore_index=True)
        df_metrics = df_metrics.append({'epoch':epoch,'Train_loss':metrics_summary['Train_loss'],'Valid_loss':metrics_summary['Valid_loss']},ignore_index=True)
        df_metrics.to_csv( save_path+'metrics.csv', index=False)
        df_values.to_csv(save_path+'values.csv', index=False)
        if phase == 'Valid' and loss < best_loss:
            best_loss = loss.item()
            save_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, save_path+'.pt')
            best_model_changed = True

            print('Saving model weights as :',weights_name+'pt')

    time_elapsed = time.time() - since
    if best_model_changed:
        #copy csv files to best model folder
        shutil.copy(save_path+'metrics.csv', save_path+'best_model/metrics.csv')
        shutil.copy(save_path+'values.csv', save_path+'best_model/values.csv')


    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    #model.load_state_dict(best_model_wts)
    
    return best_loss,save_epoch

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