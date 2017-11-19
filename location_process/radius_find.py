# -*- coding:utf-8 -*-
'''
Created on 2017年11月14日

@author: 任胜勇

这是一个关于清晰经纬度的数据
现在有两种方法
现有的数据是每个商铺对应用户的id,经纬度
1.
    用用户的经纬度找一个二维平面的最高区间，保留边界用户，删除用户
2.
    以商铺的经纬度为中心画圆，寻找一个最合适的半径作为范围划分
3.
    计算信息与商店的距离，去掉一定数目距离的大值（认为异常）
我采用了第三种方法
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


# 含有shop对应用户数据的pd文件路径
pd_path = r'G:\customer&fixed-position\data\shop\person2shop/'
# 保存了shop路径的文件
npy_path = r'G:\customer&fixed-position\data\shop\shop_lc/'
shop_list = os.listdir(pd_path)

shop_lc = os.listdir(npy_path)
shop_dis = dict();
dis_sum = 0;
dis_real_sum = 0;
shop_dis_real = dict()
for sp in tqdm(shop_list):

    pr_dis = dict()
    pr_dis['dis'] = [];
    pr_dis['time_stamp'] = [];
    pr_dis['user_id'] = []
    pr_dis['longitude'] = [];
    pr_dis['latitude'] = []
    shop_lc = np.load(npy_path + sp.split('.')[0] + '.npy')
    # 获取店铺的经纬度
    s_lg = shop_lc[0];
    s_lt = shop_lc[1]
    shop_data = pd.read_csv(pd_path + sp)
    # 初始化，新增distance
    shop_data['distance_sq'] = 999
    # 计算data距离

    for _, sp_data in shop_data.iterrows():
        lg = sp_data.longitude;
        lt = sp_data.latitude
        dis = haversine(lg, lt, shop_lc[0], shop_lc[1])
        pr_dis['dis'].append(dis)
        pr_dis['time_stamp'].append(sp_data.time_stamp)
        pr_dis['user_id'].append(sp_data.user_id)
        # 经纬度有小数点记录不精确，使用字符串记录
        pr_dis['longitude'].append(str(sp_data.longitude))
        pr_dis['latitude'].append(str(sp_data.latitude))

    df = pd.DataFrame({'user_id': pr_dis['user_id'],
                       'dis': pr_dis['dis'],
                       'time_stamp': pr_dis['time_stamp'],
                       'longitude': pr_dis['longitude'],
                       'latitude': pr_dis['latitude']})
    df.to_csv(r'G:\customer&fixed-position\data\shop\shop_dis/' + sp.split('.')[0] + '.csv')



