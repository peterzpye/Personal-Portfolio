import pandas as pd
import numpy as np
import os
import tqdm

import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


input_path = '/backup/ND/dtc_digital_order_line_new/'
input_path2 = '/backup/ND/update_data/dtc order line/'
output_path = '/backup/ND/label_files/9.21/dtc_zhipeng/'

'''
----------------------------------------------------------functions------------------------------------------------------------------------
'''
def read(path):
    import pyarrow.orc as orc
    with open(path, 'rb') as file:
        data = orc.ORCFile(file)
        product = data.read().to_pandas()
        return product
    

def process_one(df, feature_lst, citymap_lst, com = True):
    if com == True:
        df = df.loc[df.REPORTING_ORDER_ORIGIN_DESC.isin(['TMALL', 'CHINA TMALL SFS'])== False].copy()
    if com == False:
        df = df.loc[df.REPORTING_ORDER_ORIGIN_DESC.isin(['TMALL', 'CHINA TMALL SFS'])].copy()
    
    df_cleaned = df.loc[(df.SHIP_CITY_NM.isnull() == False)&(df.SHIP_CITY_NM.isin(citymap_lst)), feature_lst].copy()
    df_cleaned['ORDER_DT'] = df_cleaned.ORDER_DT.apply(lambda x: x[:10])
    agg_max = df_cleaned.groupby(['SHIP_CITY_NM', 'ORDER_DT', 'UNIV_STYLE_COLOR_CD']).max()
    agg_max.columns = ['MAX_' + name for name in agg_max.columns.values]
    agg_mean = df_cleaned.groupby(['SHIP_CITY_NM', 'ORDER_DT', 'UNIV_STYLE_COLOR_CD']).mean()
    agg_mean.columns = ['MEAN_' + name for name in agg_mean.columns.values]
    agged = pd.concat([agg_max, agg_mean], axis = 1).reset_index()
    return agged

def process_all(input_path, input_path2):
        
    
    feature_lst = ['ORDER_DT', 'UNIV_STYLE_COLOR_CD','SHIP_CITY_NM',
    'DISCOUNT_PCT', 'CFP_INCL_TAX_LC', 'CPP_INCL_TAX_LC']

    citymap = pd.read_csv('/backup/POC_Data/cityMapping.csv')
    citymap_lst = citymap.City.value_counts().index.values
    city2idx = {v:k for k,v in citymap.City.to_dict().items()}
    pinyin2city = {citymap.pinyin[i]:citymap.City[i] for i in range(len(citymap)) }
    

    fnames = [name for name in os.listdir(input_path) if name[-10:] == 'snappy.csv']
    print('----------------------------processing stage------------------------------------- ')
    transformed_lst_com = []
    transformed_lst_tm = []
    
    def _transform_pinyin2city(x):
        try:
            x = str(x).replace(' ', '')
            if x[-3:] == 'SHI':
                return pinyin2city[x]
            else:
                return pinyin2city[x+'SHI']
        except:
            return x
    
    # New Data
    fnames2 = [name for name in os.listdir(input_path2) if name[-10:] == 'snappy.orc']
    for fname in tqdm(fnames2):
        df = read(input_path2 + fname)
        df['SHIP_CITY_NM'] = df['SHIP_CITY_NM'].apply(lambda x: _transform_pinyin2city(x))
        transformed_com = process_one(df, feature_lst, citymap_lst, com = True)
        transformed_tm = process_one(df, feature_lst, citymap_lst, com = False)
        transformed_lst_com.append(transformed_com)
        transformed_lst_tm.append(transformed_tm)

    #Old Data
    for fname in tqdm(fnames):
        df = pd.read_csv(input_path + fname)
        df['SHIP_CITY_NM'] = df['SHIP_CITY_NM'].apply(lambda x: _transform_pinyin2city(x))
        transformed_com = process_one(df, feature_lst, citymap_lst, com = True)
        transformed_tm = process_one(df, feature_lst, citymap_lst, com = False)
        transformed_lst_com.append(transformed_com)
        transformed_lst_tm.append(transformed_tm)
    
    
    
    com_df = pd.concat(transformed_lst_com)
    tm_df = pd.concat(transformed_lst_tm)
    
    return com_df, tm_df
    
def save_file(com_df, tm_df, output_path):
    print('----------------------------saving stage------------------------------------- ')
    citymap = pd.read_csv('/backup/POC_Data/cityMapping.csv')
    citymap_lst = citymap.City.value_counts().index.values
    city2idx = {v:k for k,v in citymap.City.to_dict().items()}
    pinyin2city = {citymap.pinyin[i]:citymap.City[i] for i in range(len(citymap)) }
    for city in tqdm(citymap_lst):
        com_to_save = com_df.loc[com_df.SHIP_CITY_NM == city].copy()
        if len(com_to_save) != 0:
            com_to_save.to_csv(output_path + str(city2idx[city]) + '_com.csv', index = False)
        tm_to_save = tm_df.loc[tm_df.SHIP_CITY_NM == city].copy()
        if len(tm_to_save) != 0:
            tm_to_save.to_csv(output_path + str(city2idx[city]) + '_tmall.csv', index = False)
    
    print('finished!')
        

def main():
    

    try:
        os.mkdir(output_path)
        print('directory made')
    except:
        print('directory already exist')
    com_df, tm_df = process_all(input_path,input_path2)
    save_file(com_df, tm_df, output_path)
main()
    