import torch.nn as nn
import torch.nn.functional as F
import torch
#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#https://www.kaggle.com/code/khalildmk/simple-two-layer-bidirectional-lstm-with-pytorch/notebook
class Net(nn.Module):
    def __init__(self,embedding_dim,input_dim,output_dim,hidden_dim,batch_size,num_layers):
        super(Net,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.conv1= nn.Conv2d(in_channels=(12,1000),kernel_size=(12,3),)
        
        #Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)
        # Define the LSTM layer
        self.lstm = eval('nn.LSTM')(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self,input):
        conv_out=self.conv1()