#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    :2018.9.8
# @Author  :wcx
# @Site    :
# @File    :
# @Software:Pycharm
import pandas as pd
import numpy as np
from hmmlearn import hmm

data = pd.read_csv('D:/test_repair_data.csv')
sample_data = data['frequency']
data_frequency_min = data['frequency'].min()
data_frequency_max = data['frequency'].max()
length = len(sample_data)
sample_data_train = sample_data[0:length-3]
sample_data_train = [[i] for i in sample_data_train]
sample_data_test = sample_data[length-3:length]
n = 20  # 这里是指隐形状态的个数，可以自由改动
lis = []
for i in range(1, 80):
    model = hmm.GaussianHMM(n_components=i, covariance_type='full')
    model_param = model.fit(sample_data_train)   # 不同的样本来进行拟合
    hmm_model_score = model.score(sample_data_train)  # 不同样本的拟合优劣的评价
    lis.append(hmm_model_score)
print lis.index(max(lis))+1














