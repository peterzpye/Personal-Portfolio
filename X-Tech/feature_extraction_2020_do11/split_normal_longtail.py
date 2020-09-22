import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil
import hashlib



def get_agged():
    label_path = '/backup/ND/label_files_yh/city_level/'
    dfs = []
    for fname in tqdm(os.listdir(label_path)):
        if fname[-4:] == '.csv':

            dfs.append(pd.read_csv(label_path+fname))
    df = pd.concat(dfs)
    agged = df.groupby('sku').agg({"qty": np.sum, "date": pd.Series.nunique})
    return agged


def get_sku_list():
    target_path = '/home/yh/11/info/sku_list.csv'
    df = pd.read_csv(target_path)
    if 'sku_list.csv' not in os.listdir('/home/yezhipeng/info/'):
        df.to_csv('/home/yezhipeng/info/sku_list.csv')
        print('copy saved to /home/yezhipeng/info/')
    return df['Prod Cd'].tolist()

def main():
    agged = get_agged()
    normal = agged[(agged.qty >= agged.qty.quantile(0.5)) & (agged.date >= 24*7)].index.values
    normal_enc = [x for x in normal]
    longtail = agged[(agged.qty >= agged.qty.quantile(0.25)) & (agged.date >= 10*7) & (agged.date <= 24*7)].index.values
    longtail_enc = [x for x in longtail if x not in normal]
    seasonal = agged[(agged.index.isin(longtail) == False) & (agged.index.isin(normal) == False)].index.values
    seasonal_enc = [x for x in seasonal]
    
    # add target
    sku_list = get_sku_list()
    for sku in sku_list:
        if sku not in longtail_enc:
            if sku not in seasonal_enc:
                normal_enc.append(sku)
    
    if 'sku_normal_longtail_split' in os.listdir('/home/yezhipeng/'):
        print('directory sku_normal_longtail_split already exists')
    else:
        
        os.mkdir('/home/yezhipeng/sku_normal_longtail_split/')
        print('directory sku_normal_longtail_split created!')
        
    dir_path = '/home/yezhipeng/sku_normal_longtail_split/'
    with open(dir_path + "longtail_sku.txt", "w") as text_file :
        for sku in tqdm(longtail_enc):
            text_file.writelines([sku, '\n'])
    with open(dir_path + "normal_sku.txt", "w") as text_file :
        for sku in tqdm(normal_enc):
            text_file.writelines([sku, '\n'])

    with open(dir_path + "seasonal_sku.txt", "w") as text_file :
        for sku in tqdm(seasonal_enc):
            text_file.writelines([sku, '\n'])
    print('saving completed!')

