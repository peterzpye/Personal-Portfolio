import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from tqdm import tqdm




city_mapping = pd.read_csv('/backup/POC_Data/cityMapping.csv')
city_mapping = city_mapping.reset_index()


def truncate(province):
    if province not in['黑龙江省', '内蒙古自治区']:
        province = province[:2]
    else:
        province = province[:3]
    return province
def get_filename(path):
    all_fname = os.listdir(path)
    target_lst = []
    for f in all_fname:
        if f.split('_')[-1] == 'feature.csv':
            target_lst.append(f)
    return target_lst
def get_com_pref():
    province = ['北京', '河北', '吉林', '辽宁', '天津', '黑龙江', '内蒙古', '甘肃', '宁夏', '山西', '青海', '四川', '重庆', '陕西', '江西', '山东', '上海', '西藏', '新疆', '河南', '安徽', '湖北', '江苏', '浙江', '福建', '云南', '广西', '广东', '海南', '贵州', '湖南'] 
    BJ = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    OS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4] 
    YD = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    YS = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3] 
    GF = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 1, 1, 1, 1, 1, 1, 1] 
    com_pref = pd.DataFrame({'province': province, 'BJ': BJ, 'OS': OS, 'YD':YD, 'YS' : YS, 'GF': GF})
    com_pref = com_pref.set_index('province')
    return com_pref
def get_tmall_pref():
    province = ['北京', '河北', '吉林', '辽宁', '天津', '黑龙江', '内蒙古', '甘肃', '宁夏', '山西', '青海', '四川', '重庆', '陕西', '江西', '山东', '上海', '西藏', '新疆', '河南', '安徽', '湖北', '江苏', '浙江', '福建', '云南', '广西', '广东', '海南', '贵州', '湖南'] 
    BJ = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5] 
    OS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4] 
    YD = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 
    YS = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3] 
    GF = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1] 
    com_pref = pd.DataFrame({'province': province, 'BJ': BJ, 'OS': OS, 'YD':YD, 'YS' : YS, 'GF': GF})
    com_pref = com_pref.set_index('province')
    return com_pref
def get_pref_dict(com_pref):
    '''
    find the sequence of preference for cities in case of hub
    '''
    province2hub = {}

    for prov in com_pref.index.values:
        pref_lst = []
        seq = com_pref.loc[prov].sort_values().index.values
        for hub in seq:
            if hub not in ['OS', 'YS']:
                pref_lst.append(hub)
        province2hub[prov] = pref_lst
    return province2hub
def get_city2hub(province2hub):
    city2hub = {}
    for city_id in city_mapping.index.values:
        city2hub[city_id] = province2hub[truncate(city_mapping.loc[city_id, 'Province'])]
    return city2hub



def get_thub():
    '''
    return BJ, YD, GF hub data constructed from city level data
    '''
    tm_path = '/backup/POC_Data/feature/city_level/tmall_file/'
    tm_city_file_names = get_filename(tm_path)
    tm_pref = get_tmall_pref()
    tm_pref_dict = get_pref_dict(tm_pref)
    city_hub = get_city2hub(tm_pref_dict)
    
    dfcs = []
    idx = 0
    for city in tm_city_file_names:

        i = 0
        name = str()
        while city[i].isdigit():
            name += city[i]
            i += 1
        dfc = pd.read_csv(tm_path + city)
        dfc['hub'] = city_hub[int(name)][0]
        dfcs.append(dfc)
        idx += 1
    onethubdf = pd.concat(dfcs, axis = 0)
    onethubdf.fillna(0,inplace = True)
    BJ = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ]).first()
    BJ['qty'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['qty'].sum()
    BJ['DISCOUNT_PCT'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['DISCOUNT_PCT'].mean()
    BJ['CFP_INCL_TAX_LC'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['CFP_INCL_TAX_LC'].mean()
    
    YD = onethubdf[onethubdf.hub == 'YD'].copy().groupby(['date', 'sku' ]).first()
    YD['qty'] = onethubdf[onethubdf.hub == 'YD'].copy().groupby(['date', 'sku' ])['qty'].sum()
    YD['DISCOUNT_PCT'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['DISCOUNT_PCT'].mean()
    YD['CFP_INCL_TAX_LC'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['CFP_INCL_TAX_LC'].mean()
    GF = onethubdf[onethubdf.hub == 'GF'].copy().groupby(['date', 'sku' ]).first()
    GF['qty'] = onethubdf[onethubdf.hub == 'GF'].copy().groupby(['date', 'sku' ])['qty'].sum()
    GF['DISCOUNT_PCT'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['DISCOUNT_PCT'].mean()
    GF['CFP_INCL_TAX_LC'] = onethubdf[onethubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['CFP_INCL_TAX_LC'].mean()
    return  BJ.reset_index(),YD.reset_index(),GF.reset_index()

def get_chub():
    '''
    return BJ, YD, GF hub data constructed from city level data
    '''
    com_path = '/backup/POC_Data/feature/city_level/com_file/'
    com_city_file_names = get_filename(com_path)
    com_pref = get_com_pref()
    com_pref_dict = get_pref_dict(com_pref)
    city_hub = get_city2hub(com_pref_dict)
    dfcs = []
    idx = 0
    for city in com_city_file_names:

        i = 0
        name = str()
        while city[i].isdigit():
            name += city[i]
            i += 1
        dfc = pd.read_csv(com_path + city)
        dfc['hub'] = city_hub[int(name)][0]
        dfcs.append(dfc)
        idx += 1
    onechubdf = pd.concat(dfcs, axis = 0)
    onechubdf.fillna(0,inplace = True)
    BJ = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ]).first()
    BJ['qty'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['qty'].sum()
    BJ['DISCOUNT_PCT'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['DISCOUNT_PCT'].mean()
    BJ['CFP_INCL_TAX_LC'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['CFP_INCL_TAX_LC'].mean()
    
    YD = onechubdf[onechubdf.hub == 'YD'].copy().groupby(['date', 'sku' ]).first()
    YD['qty'] = onechubdf[onechubdf.hub == 'YD'].copy().groupby(['date', 'sku' ])['qty'].sum()
    YD['DISCOUNT_PCT'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['DISCOUNT_PCT'].mean()
    YD['CFP_INCL_TAX_LC'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['CFP_INCL_TAX_LC'].mean()
    GF = onechubdf[onechubdf.hub == 'GF'].copy().groupby(['date', 'sku' ]).first()
    GF['qty'] = onechubdf[onechubdf.hub == 'GF'].copy().groupby(['date', 'sku' ])['qty'].sum()
    GF['DISCOUNT_PCT'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['DISCOUNT_PCT'].mean()
    GF['CFP_INCL_TAX_LC'] = onechubdf[onechubdf.hub == 'BJ'].copy().groupby(['date', 'sku' ])['CFP_INCL_TAX_LC'].mean()
    return BJ.reset_index(),YD.reset_index(),GF.reset_index()


def main():
    BJ_com_feature, YD_com_feature, GF_com_feature = get_chub()
    BJ_com_feature.to_csv('./features/BJ_com_feature.csv')
    YD_com_feature.to_csv('./features/YD_com_feature.csv')
    GF_com_feature.to_csv('./features/GF_com_feature.csv')
    print('com done!')
    BJ_tmall_feature, YD_tmall_feature, GF_tmall_feature = get_thub()
    BJ_tmall_feature.to_csv('./features/BJ_tmall_feature.csv')
    YD_tmall_feature.to_csv('./features/YD_tmall_feature.csv')
    GF_tmall_feature.to_csv('./features/GF_tmall_feature.csv')
    print('tmall done!')