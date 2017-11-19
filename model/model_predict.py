# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:16:39 2017

@author: 任胜勇

因为存在部分商场数据量过大，未完全拟合
于是需要重复训练，在这里将early_stop_num数值变小便是为了尽量服务于未训练完成的模型
"""
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import os
# the optimizer we will use
import xgboost as xgb

LOAD = True

# Define the data path ，and read it
path = 'G:\customer&fixed-position/'
save_path = r'G:\customer&fixed-position\xgb_model/'
dis_path = r'G:\customer&fixed-position\data\shop\shop_dis/'
sp_lc_path = r'G:\customer&fixed-position\data\shop\shop_lc/'
train_data_path = r'G:\customer&fixed-position\data\test_mall__/'
df = pd.read_csv(path + 'trainperson-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path + 'trainshop-ccf_first_round_shop_info.csv')
test = pd.read_csv(path + 'Btest-evaluation_public.csv')


# 获取mall的总数，方便于之后以mall为单位的训练
mall_list = list(set(list(shop.mall_id)))

# 定义结果集
result = pd.DataFrame()



for ID, mall in enumerate(mall_list):
    '''针对于部分模型训练不足的二次训练'''
    print('{} 读取 {} 商场数据'.format(datetime.now(),mall))
    train1 = pd.read_csv(train_data_path + mall + '.csv').reset_index(drop=True)

    # 获取训练集
    df_train = train1[train1.shop_id.notnull()]

    # 获取测试集
    df_test = train1[train1.shop_id.isnull()]

    # 获取标签
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))

    # 添加标签
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    # 定义种类
    num_class = df_train['label'].max() + 1

    # 定义超参数
    # 分类器的使用我不清楚
    params = {
        'objective': 'multi:softmax',
        'min_child_weight': 2,
        'gamma': 0.25,
        'subsample': 0.82,
        'colsample_bytree': 0.7,
        'eta': 0.075,
        'max_depth': 4,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': num_class,
        'scale_pos_weight': 0.8,
        'silent': 1,

    }

    # 获取特征wifi和位置
    feature = [x for x in train1.columns if
               x not in ['user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos', 'DATE', 'Unnamed: 0']]

    # 为了之后的重复训练，在这里先保存数据
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])

    # 对于训练集，我在坐标上添加一定噪音
    # df_train = noise_Add(df_train)


    '''分类器的使用'''

    # 训练部分

    ''' Q2: 关于训练集数据的准确形式
    A2：训练集的Columns是以特征wifi与精度维度组成的数据，
    值得注意的是，特征wifi并不是每一个用户都含有，这里pd很优秀的便是自动以NaN代替
    这在分类器中是可以兼容的'''

    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    num_rounds = 200
    if LOAD:
        model = xgb.Booster()
        model.load_model(save_path + mall + '.model')
        #model.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=4)
    else:
        model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=4)
    # 保存模型
    #model.save_model(save_path + mall + '.model')

    # 测试部分
    df_test['label'] = model.predict(xgbtest)
    df_test['shop_id'] = df_test['label'].apply(lambda x: lbl.inverse_transform(int(x)))
    # 记录结果
    r = df_test[['row_id', 'shop_id']]
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv(path + 'sub4.csv', index=False)

    print('{} 预测 {} 商场数据完成'.format(datetime.now(),mall))
    print('{} 第{}个商场完成\n'.format(datetime.now(), ID+1))