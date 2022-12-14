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
import matplotlib.pyplot as plt
#import neptune.new as neptune
from skopt import BayesSearchCV
from sklearn.svm import SVC
from skopt.space import Real, Categorical, Integer
import gc
gc.collect()
import os
#internal imports
from dataset import PTB_Dataset
from train import train_loop
from model import My_Network
#from predict import predict
from validation import validation_loop
from utils import accuracy_fn

#constants
from constants import REC_PATH,CSV_PATH,DATASET_LIMIT,BATCH_SIZE,N_LEADS,N_CLASSES,NEPOCHS

torch.cuda.empty_cache()
#%%

#main function
def main(weights_name,epochs=50,batch_size=64,lr = 1e-4,momentum=0.9):
    CustomDataset = PTB_Dataset(CSV_PATH,REC_PATH,transforms.ToTensor())

    #Train %
    TRAIN_SIZE = math.floor(CustomDataset.__len__()*0.75)
    print(CustomDataset.__len__(),TRAIN_SIZE)
    TEST_SIZE = CustomDataset.__len__()-TRAIN_SIZE
    train_dataset,test_dataset = torch.utils.data.random_split(CustomDataset,[TRAIN_SIZE, TEST_SIZE])
    VALID_SIZE = math.floor(TRAIN_SIZE*0.4)
    TRAIN_SIZE = TRAIN_SIZE - VALID_SIZE
    train_dataset,valid_dataset = torch.utils.data.random_split(train_dataset,[VALID_SIZE, TRAIN_SIZE])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE)
    dataloaders = {
        'Train':
            train_dataloader,
        'Valid':
            valid_dataloader,
        'Test':
            test_dataloader
    }
    set_sizes = {
        'Train':TRAIN_SIZE,
        'Valid':VALID_SIZE,
        'Test':TEST_SIZE
    }


    #print(f'test_dataset {test_dataset}')
    model = My_Network(lstm_input_dim = 10,hidden_dim = 10,num_layers = 2)
    model=model.to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    #print('a')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=momentum,weight_decay=lr/epochs)
    print(model)

    _ = train_loop(dataloaders=dataloaders,model=model,loss_fn=loss_fn,optimizer=optimizer,device=device,epochs=epochs,weights_name = weights_name,evaluate=True)

#%%
#learning rate, momentum = 0.9, decay = learning rate / epochs
def hpt(lr_list,momentum_list,epochs):
    for lr in lr_list:
        for momentum in momentum_list:
            name = 'lr' + str(lr) +"_" + 'mom' + str(momentum)
            name_pt = name+".pt"
            print(name)
            main(weights_name=name_pt,epochs=epochs,batch_size=64,lr=lr,momentum=momentum)
            model = None
            torch.cuda.empty_cache()
            gc.collect()
    print('hpt done')

#%%

epochs = 2
lr_list =[1e-2,1e-3,1e-4]
momentum_list = [0.9,0.8,0.7]


<<<<<<< HEAD
#valami=hpt(lr_list=lr_list,momentum_list=momentum_list,epochs=epochs)
#print(valami)
=======
valami=hpt(lr_list=lr_list,momentum_list=momentum_list,epochs=epochs)
print(valami)
>>>>>>> 859bb1da9020ce72d3e50248cefc3e7f8c7034d8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('./results/best_model.pt')
model.to(device)

def plot_results(model,dataloader,device,dir):
    pred_history=[]
    true_history=[]
    for batch, (input,target) in enumerate(DataLoader):
        outputs = model(input)
        pred_history = pred_history + outputs.detach().cpu().numpy().tolist()
        true_history = true_history + target.detach().cpu().numpy().tolist()
    
    #visualize things
    print('Used model:')
    print(model)

    #visualize train and valdiaiton loss:
    path='results/a'
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    













#%%
"""
print("show losses per epochs")
x=range(0,NEPOCHS)
plt.plot(x,train_loss,label='train loss')
plt.plot(x,valid_loss,label='validation loss')
plt.legend(loc='upper right')

plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.title('Train and validation losses vs epochs')
plt.savefig('loss.png')
plt.show()
plt.close()"""
#%%

















"""
print("train_loss_history:")
x=range(0,NEPOCHS)
fig, ax = plt.subplots()
plt.scatter(x,train_loss_history,label='loss')
plt.savefig('loss.png')
plt.show()
plt.close()
print(train_loss_history)
#print(len(train_output_history))
#print(len(train_true_history))

print("last train values")
#x=range(0,len(train_output_history))
plt.scatter(train_true_history,train_output_history,color='blue',label='x:true, y: prediction')
#plt.scatter(x,train_true_history,color='red',label='true')
plt.show()
plt.savefig('train.png')
plt.close()


#x=range(0,TEST_SIZE)
#plot vlaidation data:
plt.scatter(valid_true_history,valid_pred_history,color='blue',label='x:true, y: prediction')
#plt.scatter(x,valid_true_history,color='red',label='true')
plt.legend()
plt.savefig('validation.png')
plt.show()
plt.close()
# %%
#save to file
f = open('valid_results.csv',"w")
f.write('pred;target\n')
for i in range(len(valid_pred_history)):
    #print(str(valid_pred_history[i])+';'+str(valid_true_history[i])+'\n')
    f.write(str(valid_pred_history[i])+";"+str(valid_true_history[i])+"\n")
f.close()
"""

