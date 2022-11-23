
#%%
import pandas as pd
import os

from utils import load_single_raw_data, calc_BMI,reset_sex
from constants import REC_PATH, SAMPLING_RATE,DATASET_LIMIT
from torchvision import transforms


#%%
class PTB_Dataset():
    def __init__(self,csv_file,rec_dir,transform):
        """
        Custom dataset for PTB-XL dataset
        Args:
            csv_file (string): path to ptbxl_database.csv
            rec_dir (string): path to the ECG recordings:
        Transformations:
            ToTensor():
        """
        from constants import DATASET_LIMIT,DATASET_TYPE
        #print('a')
        #print(isinstance(csv_file,str))
        if DATASET_TYPE == 'BMI':
            self.data = pd.read_csv(csv_file,header=0,usecols=['ecg_id','weight','height','filename_lr'])  #names=['ecg_id','weight','height','filename_lr'])
            #print(self.data)
            #add BMI col
            self.data = calc_BMI(self.data)
        elif DATASET_TYPE == 'SEX':
            self.data = pd.read_csv(csv_file,header=0,usecols=['ecg_id','sex','filename_lr'])  #names=['ecg_id','weight',]
            self.data = reset_sex(self.data)
        #print(self.data)
            
        self.rec_dir = rec_dir
        self.transform = transform
    
    def __len__(self):
        from constants import DATASET_TYPE
        if DATASET_TYPE == 'BMI':
            return len(self.data['BMI'])
        elif DATASET_TYPE == 'SEX':
            return len(self.data['sex'])
    
    def __getitem__(self,idx):
        from constants import DATASET_TYPE
        if DATASET_TYPE == 'BMI':
            label = self.data['BMI'].iloc[idx]
        elif DATASET_TYPE == 'SEX':
            label = self.data['sex'].iloc[idx]
        #print(label)
        """print(self.rec_dir)
        print(self.data['filename_lr'].iloc[idx])
        """
        rec = load_single_raw_data(self.data['filename_lr'].iloc[idx],SAMPLING_RATE,REC_PATH)
        #print(rec.shape)
        
        #if idx == 1: print('rec:',rec)
        
        rec_as_tensor = self.transform(rec.astype(float))
        #print(f'label type: {label.dtype}')
        #label = label.float()
        #print("get",rec_as_tensor.shape)
        ################################
        # might be some transformations here...
        ################################

        return rec_as_tensor, label
    




# %%
