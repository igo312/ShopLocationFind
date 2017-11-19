import numpy as np 
import pandas as pd 
import datetime
from tqdm import tqdm 


person_infos = pd.read_csv(r'G:\customer&fixed-position/trainperson-ccf_first_round_user_shop_behavior.csv')
data = []
def get_time(time_stamp):
    date = time_stamp.split(' ')[0]
    tm = time_stamp.split(' ')[1]
    date = [buff for buff in date.split('-')]
    tm = [buff for buff in tm.split(':')]
    
    # Get the weekday 
    date = datetime.datetime(int(date[0]),int(date[1]),int(date[2])).strftime('%w')
    date = int(date)
    return [date, int(tm[0]), int(tm[1])]

for index in tqdm(range(person_infos.shape[0])):
    person = person_infos[index:index+1]
    time_stamp = person.time_stamp.values[0]
    tm = get_time(time_stamp)
    data.append(tm)
    
data = np.asarray(data)
np.save(r'G:\customer&fixed-position\data\npy_file/time_data.npy',data)
