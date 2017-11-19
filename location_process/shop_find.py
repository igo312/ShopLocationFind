# -*- coding:utf-8 -*-
import pandas as pd
from tqdm import tqdm
import numpy as np 
import os

path = r'G:\customer&fixed-position/'
save_path=r'G:\customer&fixed-position\data\shop\person2shop/'
person = pd.read_csv(path+'trainperson-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path+'trainshop-ccf_first_round_shop_info.csv')
test = pd.read_csv(path+'ABtest-evaluation_public.csv')
# 商场映射到person 
data_total = pd.merge(person,shop[['mall_id','shop_id']],how='left',on='shop_id')

data_total = pd.concat([data_total,test])
shop_list = list(set(list(shop.shop_id)))
print(len(shop_list))
#shop_list = os.listdir(r'G:\customer&fixed-position\data\shop\buff/')
for sp in tqdm(shop_list):
    sp=sp.split('.')[0]
    data = data_total[data_total.shop_id==sp]
    lg = data.longitude
    lt = data.latitude
    user_id = data.user_id
    time_stamp = data.time_stamp
    df = pd.DataFrame({
        'user_id':user_id.values,
        'longitude':lg.values,
        'latitude':lt.values,
        'time_stamp': time_stamp.values
        })
    df.to_csv(save_path+sp+'.csv',index=False)
    
    shop_lg = shop[shop.shop_id==sp].longitude.values[0]
    shop_lt = shop[shop.shop_id==sp].latitude.values[0]
    shop_lc = np.asarray([shop_lg,shop_lt])
    np.save(r'G:\customer&fixed-position\data\shop\shop_lc/'+sp+'.npy',shop_lc)