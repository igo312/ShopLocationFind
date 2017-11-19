import pandas as pd 
import numpy as np 

data_path = r'F:\datasets\customer&fixed-position/'
save_path = r'F:\datasets\customer&fixed-position/'
train_data = pd.read_csv(data_path+'trainperson-ccf_first_round_user_shop_behavior.csv')
train_time = pd.read_csv(data_path+'time_train.csv')
train = pd.merge(train_data,train_time,how='left',on='user_id')

test_data = pd.read_csv(data_path+'ABtest-evaluation_public.csv')
test_time = pd.read_csv(data_path+'time_test.csv')
test = pd.merge(test_data,test_time,how='left',on='user_id')

train.to_csv(data_path+'with_time_train.csv')
test.to_csv(data_path+'with_time_test.csv')
print('completed.')


