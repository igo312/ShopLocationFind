#-*- coding: utf-8 -*-
'''
    这里我将只才用wifi信号去实现定位，
    去获取最优的wifi选择区间。
    同时学习lightlgb 的基础使用
'''

import numpy as np 
import pandas as pd 
import lightgbm as lgb 
from wifi_select import wifi_select
import os 
from datetime import datetime
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.grid_search import GridSearchCV
import random
'''
    在寻找最优解时应该使迭代次数尽量少（或者训练集变大）
    并且使用同一个训练集
    并且在确定范围时，应该可以进行提前的测试，粗调，大概的系数，不要从头开始
    考虑计算成本
'''
#定义欲训练的特征
feature=['longitude','user_id','latitude','time_stamp','wifi_infos','mall_id','shop_id','Unnamed: 0']

save_path=r'G:\customer&fixed-position\data\wifi\wifi_count/'
val_path=r'G:\customer&fixed-position\data\val_data/'

# 这是关于去除频次太低的wifi的阈值
threshold=15
# 这是关于保留强度前几个的值
remain_num=3
# 这个关于删除公共wifi几个的值
delate_rate=0.1



# 获取测试集的信息(以商场为单位)
mall_list = os.listdir(val_path)
iter_num=1

best_params={
    'threshold':-1,
    'remain_num':-1,
    'delate_rate':-1,
    'low_error':100
    }

'''
    根据程序，delate_rate放最底层，
    remain_num放最外面，
    threshold最中间
'''

for remain_num in range(2,5,1):
    # 注意对于寻找最优解应该用同一个测试集
    malls=[]
    malls.append(mall_list[0])
    malls.append(mall_list[2])
    malls.append(mall_list[4])

    # 为了方便读取数据在这里用较小检验
    # 读取数据
    train_data=pd.read_csv(val_path+malls[0])
    train_data=pd.concat([train_data,pd.read_csv(val_path+malls[1])])
    train_data=pd.concat([train_data,pd.read_csv(val_path+malls[2])])



    # 初始化记录wifi频次的字典
    wifi_dict=dict()
    # 初始化连接的wifi
    wifi_con_remain=[]
    # 初始化wifi信号前三列表
    wifi_remain=[]
    # 初始化一个变量保存数据
    l=[]
    for index,row in tqdm(train_data.iterrows()):
        # 分离数据,注意这里的wifi_list只是针对一个用户
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]

        # 判断是否连接
        status=0
        for i in wifi_list:
            '''
                        这个循环的目的是：
                1.分离wifi信息，往用户额外加上信号名称与强度        
                2.获取以信号名称和频次的字典        
            '''

            # 这里将在用户原先信息的基础上加上
            # 元素名为wifi名称，元素值为信号强度的信息
            row[i[0]] = int(i[1])

            # 记录wifi出现的次数
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1

            # 存储连接的wifi信号
            if i[2] == 'true':
                wifi_con_remain.append(i[0])
                status=1
        if status==0:
            wifi_con_remain.append('NONE')

        # 保留wifi信息（两个）
        if len(wifi_list) < 4:
            wifi_remain.append('NONE')
        else:
            sort_list = sorted(wifi_list, key=lambda wifi_infos: int(wifi_infos[1]), reverse=True)
            save_wifi = sort_list[:remain_num]
            wifi_remain.append(row[[wifi_name[0] for wifi_name in save_wifi]])

        # 记录数据
        l.append(row)

    #delate_rate = 0.0
    for threshold in range(4):
        threshold += 19
        print('{} 第{}次迭代开始'.format(datetime.now(), iter_num))

        # 获取认为的公共wifi名称
        #wifi_delate_name = wifi_select(save_path,delate_rate)

        '''目的：分辨低于阈值的wifi信号（认为没有作用）'''
        # 这个循环获取到了低于阈值的wifi名称
        delate_wifi = []
        #delate_wifi.extend(list(wifi_delate_name))
        for i in wifi_dict:
            if wifi_dict[i] < threshold:
                delate_wifi.append(i)
        #delate_wifi=list(set(delate_wifi))

        # 这个循环获取到了高于阈值的wifi名称
        # 并以此判断筛选各个用户中的信息
        # 注意key中仍含有位置等等信息，只是排除了不需要的wifi信号
        m = []
        for index, row in enumerate(l):
            new = {}
            for n in row.keys():
                if n not in delate_wifi:
                    new[n] = row[n]

            # 添加欲保留的wifi信息
            if type(wifi_remain[index] == 'NONE') == bool:
                pass
            else:
                remain_dict = dict(zip(wifi_remain[index].keys(), wifi_remain[index].values))
                for n in remain_dict.keys():
                    new[n] = remain_dict[n]

            # 如果连接的wifi高于阈值，可能其为公共wifi没有代表意义
            try:
                if type(wifi_con_remain[index] == 'NONE') == bool:
                    pass
                else:
            #        if wifi_con_remain[index] not in list(wifi_delate_name):
            #            new[wifi_con_remain[index]] = -15
                    new[wifi_con_remain[index]] = -15
            except:pass

            # 记录数据
            m.append(new)

        # 获取处理后的最终数据集
        train = pd.DataFrame(m)
        # 筛选特征，这里只选wifi
        df_train = train[[x for x in train.columns if
                          x not in feature]]
        # 填充缺省值
        df_train=df_train.fillna(-999.)

        lbl=preprocessing.LabelEncoder()
        df_label = lbl.fit_transform(train['shop_id'])


        # 获取到最终的训练集
        #lgb_data=lgb.Dataset(df_train,df_label)
        num_val = int(df_train.shape[0]*0.2)
        lgb_train=lgb.Dataset(df_train,df_label)
        lgb_eval=lgb.Dataset(df_train[:num_val],df_label[:num_val],reference=lgb_train)

        # 获取到标签数目
        num_class=df_label.max()+1

        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric':  'multi_error',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_class':num_class
        }
        model=lgb.train(params,
                  lgb_train,
                  valid_sets=lgb_eval,
                  num_boost_round=6
                  )
        buff_score = list(model.best_score.values())
        buff_score = buff_score[0]['multi_error']
        if best_params['low_error']>buff_score:
            best_params['threshold']=threshold
            best_params['remain_num']=remain_num
            best_params['delate_rate']=delate_rate
            best_params['low_error']=buff_score

        print('{} 第{}次迭代结束'.format(datetime.now(),iter_num))
        print('现在最优参数为{}'.format(best_params))
        print('现在的参数为threshold:{}  delate_rate:{}  remain_num:{}'.format(threshold,delate_rate,remain_num))
        iter_num+=1
print('最好的参数为{}'.format(best_params))