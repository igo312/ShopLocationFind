# -*- coding: utf-8 -*-
'''
    这是关于wifi剔除的程序
    目标是剔除公共wifi
    对于这个问题，可能适用网格搜索是一个好方法
    但是可以先给定百分制
'''

import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm

#wifi_save_path=r'G:\customer&fixed-position\data\wifi\wifi_count/'
def wifi_select(wifi_save_path,select_rate):
    mall_wifi_list = os.listdir(wifi_save_path)
    name_list=[] 
    for name in tqdm(mall_wifi_list):
        wifi_buff = pd.read_csv(wifi_save_path+name)
        wifi_buff=wifi_buff.sort_values(by=['count'],ascending=False)
        dnum=int(wifi_buff.shape[0]*select_rate)
        delate_name = list(wifi_buff[:dnum].wifi_name.values)
        name_list.extend(delate_name)
    
    name_list = np.asarray(name_list)
    return name_list
#np.save(r'G:\customer&fixed-position\data\wifi/wifi_delate_name.npy',name_list)