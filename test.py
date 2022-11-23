#%%
#from model_test import My_unittest
from model import My_Network
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size = 10 #?? conv out is (4,10,1,50)
num_layers = 1 #1 do we have to stack more layer?
hidden_dim = 20 #??
num_classes = 10
batch_size = 4
#%%
#net=Network(input_size,hidden_dim,batch_size,num_layers,num_classes).to(device)
#print(net)
#conv1= torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(12,20),stride=(1,20))
#%%
test_input = torch.randn(batch_size,50,10)
#output=net(test_input)
lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
output,(h,c)=lstm(test_input)
h_reshaped=torch.reshape(h,(-1,2*hidden_dim))
#torch.nn.Linear(hidden_dim*2,num_classes)
print(output[:,-1,:].shape)
print(h_reshaped.shape)
fc1 = torch.nn.Linear(hidden_dim*2, 5)

fc2 = torch.nn.Linear(5,1)


"""if input is (N,12,12,1000) it runs, but not if it looks like (N,12,1,1000), which should be our real input
if input is (N,12,12,1000), then output=net(input).shape = (4,10,1,50) after conv2d

"""
"""
lstm input (batch_size,sequence length, input size) = 4,50,10
view()
"""
#%%
test_input = torch.randn(4,1,12,1000)
conv1= torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(12,3),stride=(1,20))
output=conv1(test_input)
reshaped = torch.movedim(output,(0,1,3),(0,2,1))
reshaped = torch.squeeze(reshaped,dim=3) 
print('conv:',output.shape)
print('conv:',reshaped.shape)
lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
output,(h,c)=lstm(reshaped)
h_reshaped=torch.reshape(h,(-1,2*hidden_dim))
#torch.nn.Linear(hidden_dim*2,num_classes)
print(h.shape)
print('lstm reshaped:',h_reshaped.shape)
#relu
h_reshaped=F.relu(h_reshaped)
fc = torch.nn.Linear(hidden_dim*2, 1)
lin_out=fc(h_reshaped)
print('lin_out:',lin_out.shape)
print('lin_out:',lin_out)
# TODO: sanity check the reshape with 0s and 1s

# %%
import torch
import numpy as np
import pandas as pd

t = torch.tensor([[1,2],[3,4]]) #dummy data
print(t)
t_np = t.numpy() #convert to Numpy array
print(t_np)
df = pd.DataFrame(t_np) #convert to a dataframe
print(df)
df.to_csv("testfile",index=False,header=False) #save to file

#Then, to reload:
df = pd.read_csv("testfile")



# %%
