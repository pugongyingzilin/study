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
amplitude_price = ((data['max_p'] - data['min_p'])[1:]).reset_index(drop=True)
amplitude_price.columns = ['amplitude_price']
# 最高价与最低价的差
amplitude_price_min = amplitude_price.min()
amplitude_price_max = amplitude_price.max()
amplitude_price_range = int((amplitude_price_max - amplitude_price_min) / 0.1 + 1)
volumn = ((data['turnover'])[1:]).reset_index(drop=True)  # 成交量
close_price = ((data['close'])[1:]).reset_index(drop=True)  # 开盘价
close_price_max = close_price.max()
close_price_min = close_price.min()
close_price_range = int((close_price_max - close_price_min)/0.1+1)
diff_price = np.diff(data['open'])  # 涨跌值
diff_price_max = diff_price.max()
diff_price_min = diff_price.min()
diff_price_range = int((diff_price_max - diff_price_min) / 0.1 + 1)
hmm_data = pd.DataFrame({'close_price': close_price, 'amplitude_price': amplitude_price, 'diff_price': diff_price})
sample_train = hmm_data.loc[0:len(volumn)-15, :]
sample_predict = hmm_data.loc[len(volumn)-14:len(volumn)-1, :]
sample_number = len(volumn)
n = 20  # 这里是指隐形状态的个数，可以自由改动
model = hmm.GaussianHMM(n_components=n, covariance_type='full')
model_param = model.fit(sample_train)  # 不同的样本来进行拟合
hmm_model_score = model.score(sample_train)  # 不同样本的拟合优劣的评价
open_vector = np.linspace(close_price_min, close_price_max, close_price_range, endpoint=True)
amplitude_vector = np.linspace(amplitude_price_min, amplitude_price_max, amplitude_price_range, endpoint=True)
diff_vector = np.linspace(diff_price_min, diff_price_max, diff_price_range, endpoint=True)
open_sample = pd.Series(np.repeat(open_vector, diff_price_range * amplitude_price_range))
amplitude_sample = pd.Series(np.tile(np.repeat(amplitude_vector, diff_price_range), close_price_range))
diff_sample = pd.Series(np.tile(diff_vector, amplitude_price_range * close_price_range))
predict_data = pd.DataFrame({'close_price': open_sample, 'amplitude_price': amplitude_sample, 'diff_price': diff_sample})
# score_list = []
# for k in range(len(diff_sample)):
#     every_list_score = model.score(sample_train.append(predict_data.iloc[k]))
#     score_list.append(every_list_score)
# print predict_data.iloc[score_list.index(max(score_list))]
print(hmm_model_score)
print(sample_predict)












