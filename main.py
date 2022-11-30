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
import neptune.new as neptune

#internal imports
from dataset import PTB_Dataset
from train import train_loop
from model import My_Network
#from predict import predict
from validation import validation_loop
from utils import accuracy_fn

#constants
from constants import REC_PATH,CSV_PATH,DATASET_LIMIT,BATCH_SIZE,N_LEADS,N_CLASSES,NEPOCHS
#%%

#
CustomDataset = PTB_Dataset(CSV_PATH,REC_PATH,transforms.ToTensor())
#print(CustomDataset.__len__())
################################################################
#Init neptune
################################################################
"""run = neptune.init(
    project="NTLAB/test",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNGRjNDgzOC04OTk5LTQ0YTktYjQ4Ny1hMTE4NzRjNjBiM2EifQ==",
)  # your credentials"""

#Train %
TRAIN_SIZE= math.floor(CustomDataset.__len__()*0.75)
TEST_SIZE=CustomDataset.__len__()-TRAIN_SIZE
train_dataset,test_dataset = torch.utils.data.random_split(CustomDataset,[TRAIN_SIZE, TEST_SIZE])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE)
#%%
#print(f'test_dataset {test_dataset}')
model = My_Network(lstm_input_dim = 10,hidden_dim = 10,num_layers = 2)
model=model.to(device)
loss_fn = nn.MSELoss()
#print('a')
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%%
train_loss=np.empty(NEPOCHS)
valid_loss=np.empty(NEPOCHS)
for iEpoch in range(NEPOCHS):
    print(f"Current epoch is [{iEpoch}/{NEPOCHS}]")
    train_preds,train_target,train_loss[iEpoch] = train_loop(train_dataloader=train_dataloader,model=model,loss_fn=loss_fn,optimizer=optimizer,device=device,size=TRAIN_SIZE,)
    valid_preds,valid_target,valid_loss[iEpoch] = validation_loop(valid_dataloader=test_dataloader,model=model,loss_fn=loss_fn,device=device)
    #print(train_loss)
    #train_acc = accuracy_fn(train_preds,train_target)
    #valid_acc = accuracy_fn(valid_preds,valid_target)
    #print(f"train accuracy: {train_acc}, valid accuracy: {valid_acc} at {iEpoch}/{NEPOCHS}")
    
#%%

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
plt.close()
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

