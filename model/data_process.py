# -- coding utf-8 --

Created on Sat Oct 28 161639 2017

@author 任胜勇
真的好辛苦
现在已经凌晨三点了
希望这个训练可以有好的结果
虽然确实是我犯了很多错，
但这就是心血了
嘻嘻

from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import os
# the optimizer we will use
import xgboost as xgb


LOAD = False


# Define the data path ，and read it
path = 'Gcustomer&fixed-position'
save_path = r'Gcustomer&fixed-positionxgb_model'
dis_path = r'Gcustomer&fixed-positiondatashopshop_dis'
sp_lc_path=r'Gcustomer&fixed-positiondatashopshop_lc'
df = pd.read_csv(path + 'trainperson-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path + 'trainshop-ccf_first_round_shop_info.csv')
test = pd.read_csv(path + 'Btest-evaluation_public.csv')

# 将商场名称映射到训练集中
df = pd.merge(df, shop[['shop_id', 'mall_id']], how='left', on='shop_id')

# 将训练集和测试机合并
'''Q1 存在数量不平均
   A1使成为NaN值存储'''
train = df

# 获取mall的总数，方便于之后以mall为单位的训练
mall_list = list(set(list(shop.mall_id)))

# 定义结果集
result = pd.DataFrame()

# 参数的设定
drop_num = 115;
threshold = 20;
remain_num = 2

# 排除已训练好的商场
mall_already=os.listdir(r'Gcustomer&fixed-positiondatatest_mall__')
mall_already=[mall.split('.')[0] for mall in mall_already]
mall_list=[mall for mall in mall_list if mall not in mall_already]
for ID, mall in enumerate(mall_list)
    print('读取 {} 商场数据'.format(mall))
    train1 = train[train.mall_id == mall].reset_index(drop=True)
    print('现在train1的shape为{}'.format(train1.shape))
    '''这里是关于经纬度的处理，利用距离处理掉一些异常值
        注意某些商店对应的训练集特别小，怎么去扩大训练集还是需要考虑的'''
    buff_df = 0
    # 获取其中的shop
    shop_list = []
    shop_list.extend(list(shop[shop.mall_id == mall].shop_id.values))
    # 删除认为的异常值
    for shop_name in shop_list
        sp_lc=np.load(sp_lc_path+shop_name+'.npy');sp_lg=sp_lc[0];sp_lt=sp_lc[1]
        dis_data = pd.read_csv(dis_path + shop_name + '.csv').reset_index()
        shop_train = train1[train1.shop_id == shop_name].reset_index(drop=True)
        shop_train = shop_train.reset_index()
        if type(buff_df) == int
            buff_df = pd.merge(shop_train, dis_data[['dis','index']],
                               how='left', on=['index'])
            buff_df = buff_df.sort_values(['dis'], ascending=False)
            # 进行处理
            remain = buff_df[buff_df.dis  drop_num]
            delate = buff_df[buff_df.dis = drop_num]
            delate.longitude=sp_lg; delate.latitude=sp_lt
            for index in range(delate.shape[0])
                delate[indexindex+1].longitude += np.random.randn()120
                delate[indexindex+1].latitude += np.random.rand()150
            buff_df = pd.concat([remain,delate])
        else
            buff = pd.merge(shop_train, dis_data[['dis','index']],
                               how='left', on=['index'])
            buff = buff.sort_values(['dis'], ascending=False)
            remain = buff[buff.dis = drop_num]
            delate = buff[buff.dis = drop_num]
            delate.longitude=sp_lg; delate.latitude=sp_lt
            for index in range(delate.shape[0])
                delate[indexindex+1].longitude += np.random.randn()120
                delate[indexindex+1].latitude += np.random.rand()150
            buff = pd.concat([remain,delate])
            buff_df = pd.concat([buff_df, buff])
    print('现在buff_df的shape为{}'.format(buff_df.shape))
    train1 = pd.concat([buff_df, test[test.mall_id==mall]])

    # 定义存储变量
    l = [];
    wifi_remain = []
    wifi_dict = {}
    wifi_con_remain = []
    # 以一行一行的顺序读取数据。
    for index, row in tqdm(train1.iterrows())
        # 分离数据
        wifi_list = [wifi.split('') for wifi in row['wifi_infos'].split(';')]

        # 判断是否连接
        status = 0

        for i in wifi_list

            # 这里将在用户原先信息的基础上加上
            # 元素名为wifi名称，元素值为信号强度的信息
            row[i[0]] = int(i[1])

            # 记录wifi出现的次数
            if i[0] not in wifi_dict
                wifi_dict[i[0]] = 1
            else
                wifi_dict[i[0]] += 1

            # 存储连接的wifi信号
            if i[2] == 'true'
                wifi_con_remain.append(i[0])
                status = 1

        if status == 0
            wifi_con_remain.append('NONE')

        # 保留wifi信息
        if len(wifi_list)  3
            wifi_remain.append('NONE')
        else
            sort_list = sorted(wifi_list, key=lambda wifi_infos int(wifi_infos[1]), reverse=True)
            save_wifi = sort_list[remain_num]
            wifi_remain.append(row[[wifi_name[0] for wifi_name in save_wifi]])

        # 记录新的用户数据
        l.append(row)

    '''目的：分辨低于阈值的wifi信号（认为没有作用）'''
    # 这个循环获取到了低于阈值的wifi名称
    delate_wifi = []
    for i in wifi_dict
        if wifi_dict[i]  threshold
            delate_wifi.append(i)

    # 这个循环获取到了高于阈值的wifi名称
    # 并以此判断筛选各个用户中的信息
    # 注意key中仍含有位置等等信息，只是排除了不需要的wifi信号
    m = []
    for index, row in enumerate(l)
        new = {}
        for n in row.keys()
            if n not in delate_wifi
                new[n] = row[n]

        # 添加欲保留的wifi信息
        if type(wifi_remain[index] == 'NONE') == bool
            pass
        else
            remain_dict = dict(zip(wifi_remain[index].keys(), wifi_remain[index].values))
            for n in remain_dict.keys()
                new[n] = remain_dict[n]

                # 如果连接的wifi高于阈值，可能其为公共wifi没有代表意义,但在这里全部保留
            try
                if type(wifi_con_remain[index] == 'NONE') == bool
                    pass
                else
                    new[wifi_con_remain[index]] = -15
            except
                pass

        m.append(new)
    # 获取到了这个商店对应用户最终的数据集
    train1 = pd.DataFrame(m)

    # 获取训练集
    df_train = train1[train1.shop_id.notnull()]
    # 保留处理好的训练集
    train1.to_csv(r'Gcustomer&fixed-positiondatatest_mall__' + mall + '.csv')
    print('{} {}商场数据保存完毕'.format(datetime.now(), mall))
    # 获取测试集
    df_test = train1[train1.shop_id.isnull()]
    # 获取标签
    '''  在这里我们将商店标签限制到了对应的商场中   '''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    # 添加标签
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    # 定义种类
    num_class = df_train['label'].max() + 1

    # 定义超参数
    # 分类器的使用我不清楚
    params = {
        'objective' 'multisoftmax',
        'min_child_weight' 2,
        'gamma' 0.25,
        'subsample' 0.82,
        'colsample_bytree' 0.7,
        'eta' 0.08,
        'max_depth' 4,
        'eval_metric' 'merror',
        'seed' 0,
        'missing' -999,
        'num_class' num_class,
        'scale_pos_weight' 0.8,
        'silent' 1,

    }

    # 获取特征wifi和位置
    feature = [x for x in train1.columns if
               x not in ['user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos', 'DATE', 'Unnamed 0']]

    # 为了之后的重复训练，在这里先保存数据
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])

    # 对于训练集，我在坐标上添加一定噪音
    # df_train = noise_Add(df_train)


    '''分类器的使用'''

    # 训练部分

    ''' Q2 关于训练集数据的准确形式
    A2：训练集的Columns是以特征wifi与精度维度组成的数据，
    值得注意的是，特征wifi并不是每一个用户都含有，这里pd很优秀的便是自动以NaN代替
    这在分类器中是可以兼容的'''

    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    num_rounds = 140
    if LOAD
        model = xgb.Booster({'nthread' 4})
        model.load_model(save_path + '6')
        model.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    else
        model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=6)
    # 保存模型
    model.save_model(save_path + mall + '.model')

    # 测试部分
    df_test['label'] = model.predict(xgbtest)
    df_test['shop_id'] = df_test['label'].apply(lambda x lbl.inverse_transform(int(x)))
    # 记录结果
    r = df_test[['row_id', 'shop_id']]
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv(path + 'sub.csv', index=False)

