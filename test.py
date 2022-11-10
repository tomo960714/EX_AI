#%%
#from model_test import My_unittest
from model import Network
import torch

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
print(h.shape)
print(h_reshaped.shape)



"""if input is (N,12,12,1000) it runs, but not if it looks like (N,12,1,1000), which should be our real input
if input is (N,12,12,1000), then output=net(input).shape = (4,10,1,50) after conv2d

"""
"""
lstm input (batch_size,sequence length, input size) = 4,50,10
view()
"""
#%%
output=conv1(test_input)
reshaped = torch.movedim(output,(0,1,3),(0,2,1))
reshaped = torch.squeeze(reshaped,dim=3) 
# TODO: sanity check the reshape with 0s and 1s
print(output.shape)
print(reshaped.shape)
# %%
