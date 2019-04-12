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


data = pd.read_table('D:/SH600000.txt', header='infer', delim_whitespace=True)
# data_time:日期   open:开盘   max_p:最高  min_p:最低
#  close:收盘价  turnover:成交量  turnover_transaction:成交额
amplitude_price = ((data['max_p'] - data['min_p'])[1:]).reset_index(drop=True).rename(
        columns={'0': 'amplitude_price'}, inplace=True)
# 每天的最高价与最低价的差
amplitude_price_min = amplitude_price.min()
amplitude_price_max = amplitude_price.max()
amplitude_price_range = int((amplitude_price_max - amplitude_price_min) / 0.01 + 1)
volumn = ((data['turnover'])[1:]).reset_index(drop=True)  # 成交量
diff_price = np.diff(data['close'])  # 涨跌值
diff_price_max = diff_price.max()
diff_price_min = diff_price.min()
diff_price_range = int((diff_price_max - diff_price_min) / 0.01 + 1)
hmm_data = pd.concat([amplitude_price, pd.DataFrame(diff_price)], axis=1)
sample_train = hmm_data[0:len(volumn)-1]
sample_predict = hmm_data.iloc[len(volumn)-1, :]
# sample.columns = ['amplitude_price',  'diff_price']
sample_number = len(volumn)
n = 8  # 这里是指隐形状态的个数，可以自由改动
model = hmm.GaussianHMM(n_components=n, covariance_type='full')
model_param = model.fit(sample_train)  # 不同的样本来进行拟合
hmm_model_score = model.score(sample_train)  # 不同样本的拟合优劣的评价
amplitude_vector = np.linspace(amplitude_price_min, amplitude_price_max, amplitude_price_range, endpoint=True)
diff_vector = np.linspace(diff_price_min, diff_price_max, diff_price_range, endpoint=True)
amplitude_sample = pd.DataFrame(np.repeat(amplitude_vector, diff_price_range))
diff_sample = pd.DataFrame(np.tile(diff_vector, amplitude_price_range))
amplitude_diff_sample = pd.concat([amplitude_sample, diff_sample], axis=1)
model_score = []
for k in range(len(diff_sample)):
    every_model_score = model.score(pd.concat([sample_train, amplitude_diff_sample.iloc[k, :]]))
    model_score.append(every_model_score)
print amplitude_diff_sample.iloc[model_score.index(max(model_score)), :]
print(hmm_model_score)
print(sample_predict)































