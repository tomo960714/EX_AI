import numpy as np
import wfdb
import ast
import torch
import pandas as pd


def load_single_raw_data(filename, sampling_rate, path):
    data = wfdb.rdsamp(path+filename) 
    #print(data)
    #print('data:',data.shape)
    data_arr = np.asarray(data[0]).transpose()
    #print(data_arr.shape)
    #print('data_arr',data_arr.shape)
    return data_arr

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    
    data = np.array([signal for signal, meta in data])
    #print(data)
    return data


def calc_BMI(df):
    """"
    Calcualting BMI based on the new Oxford formula
    1.3*weight/height^2
    """
    from constants import DATASET_LIMIT
    tmp_df = df.drop(df[df['weight'].isna() | df['height'].isna()].index)
    tmp_df['BMI'] = 1.3*tmp_df['weight']/np.power(tmp_df['height']/100,2.5)
    
    if isinstance(DATASET_LIMIT,int):
        #print('a')
        tmp_df = tmp_df[:DATASET_LIMIT]
    elif not isinstance(DATASET_LIMIT,str):
        x = input('Wrong Dataset limit, give a new integer or ALL!\n')
        if not isinstance(x,str):
            DATASET_LIMIT = int(x)
            tmp_df = tmp_df[:DATASET_LIMIT]
        else:
            DATASET_LIMIT = 'ALL'
            #No limit
    #else:
        #No limit, Nothing happens
    
    tmp_df.reset_index(drop=True, inplace=True)
    #print(tmp_df)

    return tmp_df
def reset_sex(df):
    from constants import DATASET_LIMIT
    tmp_df = df.drop(df[df['sex'].isna()].index)
    if isinstance(DATASET_LIMIT,int):
        #print('a')
        tmp_df = tmp_df[:DATASET_LIMIT]
    elif not isinstance(DATASET_LIMIT,str):
        x = input('Wrong Dataset limit, give a new integer or ALL!\n')
        if not isinstance(x,str):
            DATASET_LIMIT = int(x)
            tmp_df = tmp_df[:DATASET_LIMIT]
        else:
            DATASET_LIMIT = 'ALL'
            #No limit
    #else:
        #No limit, Nothing happens
    
    tmp_df.reset_index(drop=True, inplace=True)
    return tmp_df
def tensor2list(tensor):
    temp = tensor.detach().cpu().numpy()
    arr=np.zeros([len(tensor)])
    for i in range(len(tensor)):
        arr[i]=temp[i,]
    
    return arr
def tensor2file(tensor):
    t_np=tensor.numpy() #convert to np
    print('tensor to np shape:',t_np.shape)
    #df = pd.DataFrame(t_np) #convert to DataFrame
    #df.to_csv("testfile",index=False,header=False) #save to file

def accuracy_fn(predicted,target):
    correct = torch.sum(predicted == target)
    return correct/len(predicted)