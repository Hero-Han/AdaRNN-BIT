#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Apple, Inc. All Rights Reserved 
#
# @Time    : 2022/3/29 14:46
# @Author  : SeptKing
# @Email   : WJH0923@mail.dlut.edu.cn
# @File    : prepare_QX.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
import seaborn as sns
import os
from sqlalchemy import create_engine

data_dir = r'/Volumes/王九和/科研/农业大数据相关/实验/实验程序/deep_mulitadann/weather_data'
filepath = os.path.join(data_dir, 'QXSJ.csv')
d = open(filepath,encoding='utf-8')
df = pd.read_csv(filepath)
df = df.rename(columns = {'Mon':'Month'})
df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
df.drop(columns=['Province', 'Datetime', 'City', 'Station_Id_d', 'Cnty', 'EVP_Big','WIN_D_INST', 'WIN_S_INST','Year', 'Month', 'Day', 'Hour'],inplace=True)
df.loc[df['PRS'] > 1500,'PRS'] = 9999
df.loc[df['PRS_Sea'] > 1500, 'PRS_Sea'] = 9999
df.loc[df['TEM'] > 50, 'TEM'] = 9999
df.loc[df['DPT'] > 50, 'DPT'] = 9999
df.loc[df['VAP'] > 50, 'VAP'] = 9999
df.loc[df['PRE_1h'] > 1000, 'PRE_1h'] = 0 ##将异常值数据变成0
df.loc[df['WIN_D_Avg_2mi'] > 360, 'WIN_D_Avg_2mi'] = 9999
df.loc[df['WIN_S_Avg_2mi'] > 1000, 'WIN_S_Avg_2mi'] = 9999
# df.loc[df['WIN_D_INST'] > 360, 'WIN_D_INST'] = 9999
# df.loc[df['WIN_S_INST'] > 1000, 'WIN_S_INST'] = 9999
df.loc[df['GST'] > 360, 'GST'] = 9999
df.loc[df['GST_5cm'] > 360, 'GST_5cm'] = 9999
df.loc[df['GST_10cm'] > 360, 'GST_10cm'] = 9999
df.loc[df['GST_15cm'] > 360, 'GST_15cm'] = 9999
df.loc[df['GST_20cm'] > 360, 'GST_20cm'] = 9999
df.loc[df['GST_40Cm'] > 360, 'GST_40Cm'] = 9999
df.loc[df['GST_80cm'] > 360, 'GST_80cm'] = 9999
df.loc[df['GST_160cm'] > 360, 'GST_160cm'] = 9999
df.loc[df['RHU']> 1000, 'RHU'] = 9999
df.replace(9999, np.nan, inplace=True)
group = dict(list(df.groupby('Station_Name')))
print(group)
xj_df = df[df['Station_Name'] == '辛集']
wx_df = df[df['Station_Name'] == '武乡']
jt_df = df[df['Station_Name'] == '金堂']
xj_df.set_index('Timestamp', inplace=True)
wx_df.set_index('Timestamp', inplace=True)
jt_df.set_index('Timestamp', inplace=True)
xj_df = xj_df.drop(columns=['Station_Name'])
wx_df = wx_df.drop(columns=['Station_Name'])
jt_df = jt_df.drop(columns=['Station_Name'])
##print(xj_df.isnull().sum())
##print(wx_df.isnull().sum())
print(jt_df.isnull().sum())
xj_df.sort_index(inplace=True)
wx_df.sort_index(inplace=True)
jt_df.sort_index(inplace=True)

