'''
    I'll choose three(the shortest is one)wifi infos
    each person sorted by the signal intensity
    (The num less than 3 I'll repeat the data)
'''

import numpy as np 
import pandas as pd 
from tqdm import tqdm 

person_infos = pd.read_csv(r'F:\datasets\customer&fixed-position/trainperson-ccf_first_round_user_shop_behavior.csv')
shop_ID = pd.read_csv(r'F:\datasets\customer&fixed-position\data/shop_id.csv')
wifi_ID = pd.read_csv(r'F:\datasets\customer&fixed-position\data/wifi_names.csv')

wifi_dic = dict(zip(wifi_ID['wifi_names'],wifi_ID['id']))
shop_dic = dict(zip(shop_ID['shop_id'],shop_ID['shop_digit_id']))

def make_data(buff):
    # GET THE FINAL DATA
    buff_list = buff.split('|')
    wifi_id = wifi_dic[buff_list[0]]
    wifi_intensity = buff_list[1]
    if buff_list[2]=='false':wifi_con=0
    else: wifi_con=1
    return [wifi_id,wifi_intensity,wifi_con]

DATA=[]; LABEL=[]
for index in tqdm(range(person_infos.shape[0])):
    person = person_infos[index:index+1]
    # Get the label
    label = person.shop_id.values[0]
    label = shop_dic[label]
    
    # Define the data 
    data = []
    
    # Get the wifi infos 
    wf_data = list(person.wifi_infos)[0].split(';')
    sig_F = []
    # Sorted the wifi signal intensity
    for i in wf_data:
        f = int(i.split('|')[1])
        sig_F.append(f)
    sig_F = np.asarray(sig_F);sort_list = sig_F.argsort()
    sort_list = sort_list[::-1][:3]
    if sort_list.shape[0] == 1:
        sort_list = sort_list.repeat(3)
    elif sort_list.shape[0] == 2:
        sort_list = np.concatenate((sort_list,sort_list[-1].repeat(2)))
    
    buff_data = [wf_data[index] for index in sort_list]
    for buff in buff_data:
        data.append(make_data(buff))
    
    # Save the data and label 
    DATA.append(data)
    LABEL.append(label)

# Save the result
DATA = np.asarray(DATA)
LABEL = np.asarray(LABEL)
np.save(r'F:\datasets\customer&fixed-position\data/wifi_data.npy',DATA)
np.save('F:\datasets\customer&fixed-position\data/wifi_label.npy',LABEL)
