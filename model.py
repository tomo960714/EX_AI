#%%
import torch.nn as nn
import torch.nn.functional as F
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
#https://www.kaggle.com/code/khalildmk/simple-two-layer-bidirectional-lstm-with-pytorch/notebook
class My_Network(nn.Module):
    def __init__(self,hidden_dim,num_layers,lstm_input_dim): #
        super(My_Network,self).__init__()
        self.lstm_input_dim = lstm_input_dim
        self.hidden_dim =hidden_dim
        self.num_layers = num_layers
        #old style
        #self.conv1= nn.Conv1d(in_channels=12,out_channels=10,kernel_size=(10),stride=(10))
        #try 1: 2*5 kernel
        """self.conv1= nn.Conv1d(in_channels=12,out_channels=12,kernel_size=(5),stride=(5))
        self.conv2= nn.Conv1d(in_channels=12,out_channels=lstm_input_dim,kernel_size=(5),stride=(5))
        self.Batchnorm = nn.BatchNorm1d(12)
        self.pool= nn.AvgPool1d(kernel_size=2,stride=2)
        """
        #plus
                # Temporal analysis block 1
        self.conv1 = nn.Conv1d(in_channels=12,out_channels=16,kernel_size=(7),stride=(1),padding='same')
        self.batch1 = nn.BatchNorm1d(16)
        #relu
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 2
        self.conv2 = nn.Conv1d(in_channels=16,out_channels=16,kernel_size=(5),stride=(1),padding='same')
        self.batch2 = nn.BatchNorm1d(16)
        #relu
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        #temporal analysis block 3
        self.conv3 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=(5),stride=(1),padding='same')
        self.batch3 = nn.BatchNorm1d(32)
        #relu
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 4
        self.conv4 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=(5),stride=(1),padding='same')
        self.batch4 = nn.BatchNorm1d(32)
        #relu
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 5
        self.conv5 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=(5),stride=(1),padding='same')
        self.batch5 = nn.BatchNorm1d(64)
        #relu
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 6
        self.conv6 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3),stride=(1),padding='same')
        self.batch6 = nn.BatchNorm1d(64)
        #relu
        self.pool6 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 7
        self.conv7 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3),stride=(1),padding='same')
        self.batch7 = nn.BatchNorm1d(64)
        #relu
        self.pool7 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 8
        self.conv8 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3),stride=(1),padding='same')
        self.batch8 = nn.BatchNorm1d(64)
        #relu
        self.pool8 = nn.MaxPool1d(kernel_size=2)
        # Spatial analysis block 1
        self.spatial_conv1 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1,stride=1,padding='same')
        self.spatial_batch1 = nn.BatchNorm1d(128)
        #relu
        self.spatial_pool1 = nn.MaxPool1d(kernel_size=2)
        #self.spatial_flatten1= nn.Flatten()
        
        # Define the LSTM layer
        #self.lstm = eval('nn.LSTM')(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        #self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        # Define the output layer
        self.fc1 = nn.Linear(self.hidden_dim*2, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self,input):
        #Initalise the hidden state
        #h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).to(device)
        #c0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).to(device)
        #print('input_shape:',input.shape)
        """out = self.conv1(input)
        out = self.Batchnorm(out)
        out = F.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
         # Temporal analysis block 1
        """
        out = self.conv1(input)
        out = self.batch1(out)
        out = F.relu(out)
        out = self.pool1(out)

        # Temporal analysis block 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = F.relu(out)
        out = self.pool2(out)

        #temporal analysis block 3
        out = self.conv3(out)
        out = self.batch3(out)
        out = F.relu(out)
        out = self.pool3(out)

        # Temporal analysis block 4
        out = self.conv4(out)
        out = self.batch4(out)
        out = F.relu(out)
        out = self.pool4(out)

        # Temporal analysis block 5
        out = self.conv5(out)
        out = self.batch5(out)
        out = F.relu(out)
        out = self.pool5(out)

        # Temporal analysis block 6
        out = self.conv6(out)
        out = self.batch6(out)
        out = F.relu(out)
        out = self.pool6(out)

        # Temporal analysis block 7
        out = self.conv7(out)
        out = self.batch7(out)
        out = F.relu(out)
        out = self.pool7(out)

        # Temporal analysis block 8
        out = self.conv8(out)
        out = self.batch8(out)
        out = F.relu(out)
        out = self.pool8(out)
         # Spatial analysis block 1
        out = self.spatial_conv1(out)
        out = self.spatial_batch1(out)
        out = F.relu(out)
        out = self.spatial_pool1(out)
        #out = self.spatial_flatten1(out)
        #print('out_shape:',out.shape)
        reshaped=torch.transpose(out,1,2)
        lstm_out,(h,c)=self.lstm(reshaped) #(_ would be hidden_state,cell_state)
        #h_reshaped=torch.reshape(lstm_out[:,-1,:],(-1,2*self.hidden_dim))
        out = self.fc1(lstm_out[:,-1,:])
        out = F.relu(out)
        out= self.fc2(out) #only send the last hidden layer to the linear layer
        out = torch.squeeze(out)

        #print('model_out:',out)
        return out

#https://www.kaggle.com/code/bjoernjostein/age-estimation-using-cnn-on-12-lead-ecg
class My_Network2(nn.Module):
    def __init__(self):
        super(My_Network2,self).__init__()
        # Temporal analysis block 1
        self.conv1 = nn.Conv1d(in_channels=12,out_channels=16,kernel_size=(7),stride=(1),padding='same')
        self.batch1 = nn.BatchNorm1d(16)
        #relu
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 2
        self.conv2 = nn.Conv1d(in_channels=16,out_channels=16,kernel_size=(5),stride=(1),padding='same')
        self.batch2 = nn.BatchNorm1d(16)
        #relu
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        #temporal analysis block 3
        self.conv3 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=(5),stride=(1),padding='same')
        self.batch3 = nn.BatchNorm1d(32)
        #relu
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 4
        self.conv4 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=(5),stride=(1),padding='same')
        self.batch4 = nn.BatchNorm1d(32)
        #relu
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 5
        self.conv5 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=(5),stride=(1),padding='same')
        self.batch5 = nn.BatchNorm1d(64)
        #relu
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 6
        self.conv6 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3),stride=(1),padding='same')
        self.batch6 = nn.BatchNorm1d(64)
        #relu
        self.pool6 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 7
        self.conv7 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3),stride=(1),padding='same')
        self.batch7 = nn.BatchNorm1d(64)
        #relu
        self.pool7 = nn.MaxPool1d(kernel_size=2)

        # Temporal analysis block 8
        self.conv8 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=(3),stride=(1),padding='same')
        self.batch8 = nn.BatchNorm1d(64)
        #relu
        self.pool8 = nn.MaxPool1d(kernel_size=2)

        # Spatial analysis block 1
        self.spatial_conv1 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1,stride=1,padding='same')
        self.spatial_batch1 = nn.BatchNorm1d(128)
        #relu
        self.spatial_pool1 = nn.MaxPool1d(kernel_size=2)
        self.spatial_flatten1= nn.Flatten()

        # Fully connected layer 1
        self.fc_dense1 = nn.Linear(128,64)
        self.fc_batch1 = nn.BatchNorm1d(64)
        #relu
        self.fc_drop1 = nn.Dropout(0.2)

        # Fully connected layer 2
        self.fc_dense2 = nn.Linear(64,32)
        self.fc_batch2 = nn.BatchNorm1d(32)
        #relu
        self.fc_drop2 = nn.Dropout(0.2)
        
        #output layer
        self.fc_dense3 = nn.Linear(32,1)
    def forward(self,input):
        # Temporal analysis block 1
        out = self.conv1(input)
        out = self.batch1(out)
        out = F.relu(out)
        out = self.pool1(out)

        # Temporal analysis block 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = F.relu(out)
        out = self.pool2(out)

        #temporal analysis block 3
        out = self.conv3(out)
        out = self.batch3(out)
        out = F.relu(out)
        out = self.pool3(out)

        # Temporal analysis block 4
        out = self.conv4(out)
        out = self.batch4(out)
        out = F.relu(out)
        out = self.pool4(out)

        # Temporal analysis block 5
        out = self.conv5(out)
        out = self.batch5(out)
        out = F.relu(out)
        out = self.pool5(out)

        # Temporal analysis block 6
        out = self.conv6(out)
        out = self.batch6(out)
        out = F.relu(out)
        out = self.pool6(out)

        # Temporal analysis block 7
        out = self.conv7(out)
        out = self.batch7(out)
        out = F.relu(out)
        out = self.pool7(out)

        # Temporal analysis block 8
        out = self.conv8(out)
        out = self.batch8(out)
        out = F.relu(out)
        out = self.pool8(out)

        # Spatial analysis block 1
        out = self.spatial_conv1(out)
        out = self.spatial_batch1(out)
        out = F.relu(out)
        out = self.spatial_pool1(out)
        out = self.spatial_flatten1(out)

        # Fully connected layer 1
        out = self.fc_dense1(out)
        out = self.fc_batch1(out)
        out = F.relu(out)
        out = self.fc_drop1(out)

        # Fully connected layer 2
        out = self.fc_dense2(out)
        out = self.fc_batch2(out)
        out = F.relu(out)
        out = self.fc_drop2(out)
        
        #output layer
        out = self.fc_dense3(out)
        out = torch.squeeze(out)

        return out

#%%


# %%
