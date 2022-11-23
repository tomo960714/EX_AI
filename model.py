#%%
import torch.nn as nn
import torch.nn.functional as F
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#https://www.kaggle.com/code/khalildmk/simple-two-layer-bidirectional-lstm-with-pytorch/notebook
class My_Network(nn.Module):
    def __init__(self,input_dim,hidden_dim,batch_size,num_layers,num_classes): #
        super(My_Network,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim =hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.conv1= nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(12,3),stride=(1,20))
        
        
        # Define the LSTM layer
        #self.lstm = eval('nn.LSTM')(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        # Define the output layer
        self.fc1 = nn.Linear(self.hidden_dim*2, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self,input):
        #Initalise the hidden state
        #h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).to(device)
        #c0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).to(device)
        #print('input_shape:',input.shape)
        conv_out = self.conv1(input)
        reshaped = torch.movedim(conv_out,(0,1,3),(0,2,1))
        reshaped = torch.squeeze(reshaped,dim=3) 
        lstm_out,(h,c)=self.lstm(reshaped) #(_ would be hidden_state,cell_state)
        #h_reshaped=torch.reshape(lstm_out[:,-1,:],(-1,2*self.hidden_dim))
        out = self.fc1(lstm_out[:,-1,:])
        out = F.relu(out)
        out= self.fc2(out) #only send the last hidden layer to the linear layer
        out = torch.squeeze(out)

        #print('model_out:',out)
        return out
#%%


# %%
