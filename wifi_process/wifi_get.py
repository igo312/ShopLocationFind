import pandas as pd 
from tqdm import tqdm

path = r'F:\datasets\customer&fixed-position/'
save_con=r'F:\datasets\customer&fixed-position\data\wifi/wifi_con/'
save_count=r'F:\datasets\customer&fixed-position\data\wifi\wifi_count/'
person = pd.read_csv(path+'trainperson-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path+'trainshop-ccf_first_round_shop_info.csv')
test = pd.read_csv(path+'ABtest-evaluation_public.csv')
# 商场映射到person 
data_total = pd.merge(person,shop[['mall_id','shop_id']],how='left',on='shop_id')

data_total = pd.concat([data_total,test])
mall_list = list(set(list(shop.mall_id)))

i=0
for mall in mall_list:
    i+=1
    print('第{}个商场'.format(i))
    data = data_total[data_total.mall_id==mall]

    wifi_dict_con={}
    wifi_dict={}
    for _, row in tqdm(data.iterrows()):
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for wifi in wifi_list:
            if wifi[0] not in wifi_dict:
                wifi_dict[wifi[0]] = 0
            wifi_dict[wifi[0]] +=1
            
            if wifi[2] == 'true':
                if wifi[0] not in wifi_dict_con:
                    wifi_dict_con[wifi[0]] = 0
                wifi_dict_con[wifi[0]] += 1
        
        
    df_con = pd.DataFrame({
        'wifi_name':list(wifi_dict_con.keys()),
        'count':list(wifi_dict_con.values())
        })
    
    df_con.to_csv(save_con+mall+'.csv')
    
    df = pd.DataFrame({
        'wifi_name':list(wifi_dict.keys()),
        'count':list(wifi_dict.values())
        })
    df.to_csv(save_count+mall+'.csv')