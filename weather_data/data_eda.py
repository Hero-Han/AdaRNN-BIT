import pickle
import pandas as pd

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_loss = torch.zeros(1).to(device)
data_file = r'/Volumes/王九和/科研/农业大数据相关/实验/实验程序/AdaRNN-BIT/weather_data/QX_data.pkl'
data = pd.read_pickle(data_file)
print(data)
data_x = data['Xinji']
data_label_reg = data_x[1]
data_fea = data_x[0]
dc = data_fea[:,:]
print(len(data_fea))
print(len(data_label_reg))
print(data_x)
dist_mat = torch.zeros(2, 24).to(device)
file = r'/Volumes/王九和/科研/农业大数据相关/实验/实验程序/deep_mulitadann/weather_data/QXSJ.csv'
QXSJ = pd.read_csv(file)
Xinji_data = QXSJ[QXSJ['Station_Name']=='辛集']
c = Xinji_data.Year.value_counts()

DF = r'/Volumes/王九和/科研/农业大数据相关/实验/实验程序/PRSA_Data_1.pkl'
df = pd.read_pickle(DF)
Df = df['Changping']
fea = Df[0]
dd = fea[:,-1,:]
label  = Df[1]