#%%
#external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import copy
from GPUtil import showUtilization as gpu_usage
import csv
import os
import tqdm

#internal imports

#constants
from constants import REC_PATH,CSV_PATH,N_LEADS,N_CLASSES
#%%
def train_gender_loop(dataloaders,model,optimizer,loss_fn,device,weights_name,epochs,evaluate = True):
    if evaluate == True:
        phases =['Train', 'Valid']
    else:
        phases = ['Train']
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    fieldnames_metrics = ['epoch', 'Train_loss', 'Valid_loss']
