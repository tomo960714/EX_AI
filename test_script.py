#%%
#from model_test import My_unittest
from model import My_Network, My_Network2
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
conv1= torch.nn.Conv1d(in_channels=12,out_channels=12,kernel_size=(10),stride=(10))
test_input = torch.randn(64,12,1000)
output=conv1(test_input)

print(output.shape)
covn2 =torch.nn.Conv1d(in_channels=12,out_channels=10,kernel_size=(10),stride=(10))
out2=covn2(output)
lstm = torch.nn.LSTM(input_size=10, hidden_size=10, num_layers=2, batch_first=True, bidirectional=True)
print(out2.shape)
output,(h,c)=lstm(out2)
print(c.shape)

lstm = torch.nn.LSTM(input_size=10, hidden_size=10, num_layers=2, batch_first=True, bidirectional=True)
print(out2.shape)
output,(h,c)=lstm(out2)
#%%
net=My_Network2().to(device)
test_input = torch.randn(64,12,1000).to(device)
output=net(test_input)
print(output.shape)
# input_size = 10 #?? conv out is (4,10,1,50)
# num_layers = 1 #1 do we have to stack more layer?
# hidden_dim = 20 #??
# num_classes = 10
# batch_size = 4
# #%%
# #net=Network(input_size,hidden_dim,batch_size,num_layers,num_classes).to(device)
# #print(net)
# #conv1= torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(12,20),stride=(1,20))
# #%%
# test_input = torch.randn(batch_size,50,10)
# #output=net(test_input)
# lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
# output,(h,c)=lstm(test_input)
# h_reshaped=torch.reshape(h,(-1,2*hidden_dim))
# #torch.nn.Linear(hidden_dim*2,num_classes)
# print(output[:,-1,:].shape)
# print(h_reshaped.shape)
# fc1 = torch.nn.Linear(hidden_dim*2, 5)

# fc2 = torch.nn.Linear(5,1)


# """if input is (N,12,12,1000) it runs, but not if it looks like (N,12,1,1000), which should be our real input
# if input is (N,12,12,1000), then output=net(input).shape = (4,10,1,50) after conv2d

# """
# """
# lstm input (batch_size,sequence length, input size) = 4,50,10
# view()
# """
# #%%
# test_input = torch.randn(4,1,12,1000)
# conv1= torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(12,3),stride=(1,20))
# output=conv1(test_input)
# reshaped = torch.movedim(output,(0,1,3),(0,2,1))
# reshaped = torch.squeeze(reshaped,dim=3) 
# print('conv:',output.shape)
# print('conv:',reshaped.shape)
# lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
# output,(h,c)=lstm(reshaped)
# h_reshaped=torch.reshape(h,(-1,2*hidden_dim))
# #torch.nn.Linear(hidden_dim*2,num_classes)
# print(h.shape)
# print('lstm reshaped:',h_reshaped.shape)
# #relu
# h_reshaped=F.relu(h_reshaped)
# fc = torch.nn.Linear(hidden_dim*2, 1)
# lin_out=fc(h_reshaped)
# print('lin_out:',lin_out.shape)
# print('lin_out:',lin_out)
# # TODO: sanity check the reshape with 0s and 1s

# # %%
# import torch
# import numpy as np
# import pandas as pd

# t = torch.tensor([[1,2],[3,4]]) #dummy data
# print(t)
# t_np = t.numpy() #convert to Numpy array
# print(t_np)
# df = pd.DataFrame(t_np) #convert to a dataframe
# print(df)
# df.to_csv("testfile",index=False,header=False) #save to file

# #Then, to reload:
# df = pd.read_csv("testfile")



# %%
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
#%%
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'data/ptb-xl/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)
#%%

print(X[1,:,1].shape)
plt.plot(X[0,:,0])
plt.plot(X[0,:,1])
#print(X.shape)
# %%
