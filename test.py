#%%
import pandas as pd
import constants as con
import wfdb
from utils import load_single_raw_data,calc_BMI
data = pd.read_csv(con.CSV_PATH,header=0,usecols=['ecg_id','weight','height','filename_lr'])  #names=['ecg_id','weight','height','filename_lr'])

data = calc_BMI(data)
print(data)

#%%
rec = load_single_raw_data(data.filename_lr[0],con.SAMPLING_RATE,con.REC_PATH)

# %%
print(rec.shape)

# %%
