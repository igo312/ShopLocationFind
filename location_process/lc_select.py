# -*- coding: utf-8 -*-
'''
Created on 2017年11月15日

@author: 任胜勇

这是一个关于挑选半径的思想，
我认为距离过长的数据为异常值，我将作为异常值剔除，至于剔除多少需要跑过才知道
'''

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn import preprocessing
# the optimizer we will use
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime

path = 'G:\customer&fixed-position/'
save_path = r'G:\customer&fixed-position\xgb_model/'
save_data_path = r'G:\customer&fixed-position\xgb_data\train_withTIME/'
save_test_path = r'G:\customer&fixed-position\xgb_data\test_withTIME/'
dis_path = r'G:\customer&fixed-position\data\shop\shop_dis/'
val_path=r'G:\customer&fixed-position\data\val_data/'
df = pd.read_csv(path + 'trainperson-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path + 'trainshop-ccf_first_round_shop_info.csv')
test = pd.read_csv(path + 'ABtest-evaluation_public.csv')

# 将商场名称映射到训练集中
df = pd.merge(df, shop[['shop_id', 'mall_id']], how='left', on='shop_id')

# 将时间转换为pd形式的时间
# df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# 将训练集和测试机合并
'''Q1: 存在数量不平均
   A1:使成为NaN值存储'''
train = pd.concat([df, test])

# 获取mall的总数，方便于之后以mall为单位的训练
mall_list = list(set(list(shop.mall_id)))

# 定义结果集
result = pd.DataFrame()


# 参数设定
drop_num = 120

# 需要训练的特征
feature=['longitude','latitude']

iter_num=1
mall_list = os.listdir(val_path)
best_params={
    'drop_num':999,
    'lower_error':999
}
malls=[]
malls.append(mall_list[0])
malls.append(mall_list[2])
malls.append(mall_list[4])


# 获取测试集
train1 = train[train.mall_id == malls[0].split('.')[0]].reset_index(drop=True)
train2 = train[train.mall_id == malls[1].split('.')[0]].reset_index(drop=True)
train3 = train[train.mall_id == malls[2].split('.')[0]].reset_index(drop=True)
train1 = pd.concat([train1,train2,train3])

mall_list=mall_list = list(set(list(shop.mall_id)))
# 对于丢失值的情况进行循环测试
for drop_num in range(100,130,5):
    buff_df = 0
    for mall in tqdm(mall_list):
        #mall=mall.split('.')[0]
        '''这里是关于经纬度的处理'''
        # 获取其中的shop
        shop_list = []
        shop_list.extend(list(shop[shop.mall_id == mall].shop_id.values))
        # 删除认为的异常值
        for shop_name in shop_list:
            dis_data = pd.read_csv(dis_path + shop_name + '.csv')
            if type(buff_df) == int:
                buff_df = pd.merge(train1[train1.shop_id == shop_name], dis_data[['time_stamp', 'user_id', 'dis']],
                                   how='left', on=['time_stamp', 'user_id'])
                buff_df = buff_df.sort_values(['dis'], ascending=False)
                buff_df = buff_df[buff_df.dis < drop_num]
            else:
                buff = pd.merge(train1[train1.shop_id == shop_name], dis_data[['time_stamp', 'user_id', 'dis']], how='left',
                                on=['time_stamp', 'user_id'])
                buff = buff.sort_values(['dis'], ascending=False)
                buff = buff[buff.dis < drop_num]
                buff_df = pd.concat([buff_df, buff])
    # 处理完成 重新赋值
    train_data = buff_df
    # 删除变量

    train_data = train_data.fillna(-999.)
    print('{} 第{}次迭代开始'.format(datetime.now(), iter_num))
    df_train=train_data[feature]
    lbl = preprocessing.LabelEncoder()
    df_label = lbl.fit_transform(train_data['shop_id'])

    # 获取到最终的训练集
    # lgb_data=lgb.Dataset(df_train,df_label)
    num_val = int(df_train.shape[0] * 0.2)
    lgb_train = lgb.Dataset(df_train, df_label)
    lgb_eval = lgb.Dataset(df_train[:num_val], df_label[:num_val], reference=lgb_train)

    # 获取到标签数目
    num_class = df_label.max() + 1

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_error',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_class': num_class
    }
    model = lgb.train(params,
                      lgb_train,
                      valid_sets=lgb_eval,
                      num_boost_round=30
                      )
    buff_score = list(model.best_score.values())
    buff_score = buff_score[0]['multi_error']
    if best_params['lower_error'] > buff_score:
        best_params['drop_num']=drop_num
        best_params['lower_error']=buff_score
    print('{} 第{}次迭代结束'.format(datetime.now(), iter_num))
    print('现在最优参数为{}'.format(best_params))
    print('现在的参数为drop_num:{}'.format(drop_num))
    iter_num += 1