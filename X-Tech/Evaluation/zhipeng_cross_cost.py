import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from tqdm import tqdm


def trim(df):
#     test_date = ['2019-11-01', '2019-11-02','2019-11-03','2019-11-04','2019-11-05','2019-11-06','2019-11-07',
#           '2019-11-08','2019-11-09','2019-11-10','2019-11-11','2019-11-12','2019-11-13','2019-11-14','2019-11-15']
    test_date = ['2019-11-05','2019-11-06','2019-11-07',
            '2019-11-08','2019-11-09','2019-11-10','2019-11-11','2019-11-12','2019-11-13','2019-11-14','2019-11-15',
               '2019-11-16', '2019-11-17', '2019-11-18' ]
    return df[df.date.isin(test_date)]
def transform(df):
    

    df = trim(df)
    return df.groupby(['sku']).sum()[['qty']]

def get_onetdf():
    '''
    return all sku sales in all cities of tmall
    [index:sku, columns:cities in number, values:sales]
    '''
    tm_path = '/backup/POC_Data/feature/city_level/tmall_file/'
    tm_city_file_names = get_filename(tm_path)
    dfcs = []
    idx = 0
    for city in tm_city_file_names:

        i = 0
        name = str()
        while city[i].isdigit():
            name += city[i]
            i += 1
        dfc = pd.read_csv(tm_path + city)

        dfc = transform(dfc)

        dfc.rename(columns = {'qty': name}, inplace = True)
        dfcs.append(dfc)
        idx += 1


    onetdf = pd.concat(dfcs, axis = 1)
    onetdf.fillna(0,inplace = True)
    return  onetdf 
# onetdf = get_onetdf()
def get_onecdf():    
    '''
    return all sku sales in all cities of com
    [index:sku, columns:cities in number, values:sales]
    '''
    com_path = '/backup/POC_Data/feature/city_level/com_file/'
    com_city_file_names = get_filename(com_path)
    dfcs = []
    for city in tqdm(com_city_file_names):
        i = 0
        name = str()
        while city[i].isdigit():
            name += city[i]
            i += 1
        dfc = pd.read_csv(com_path + city)
        dfc = transform(dfc)

        dfc.rename(columns = {'qty': name}, inplace = True)
        dfcs.append(dfc)


    onecdf = pd.concat(dfcs, axis = 1)
    onecdf.fillna(0,inplace = True)
    return onecdf

'''
----------------------------------------------------get shipping preference-------------------------------------------------------------------
'''


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
def get_com_price():
    province = ['北京', '河北', '吉林', '辽宁', '天津', '黑龙江', '内蒙古', '甘肃', '宁夏', '山西', '青海', '四川', '重庆', '陕西', '江西', '山东', '上海', '西藏', '新疆', '河南', '安徽', '湖北', '江苏', '浙江', '福建', '云南', '广西', '广东', '海南', '贵州', '湖南'] 
    BJ = [10.78, 14.0, 23.0, 22.0, 15.0, 23.0, 23.0, 23.0, 23.0, 22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 19.79, 23.0, 26.0, 26.0, 17.79, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0] 
#     OS = [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 12, 26, 26, 22, 14, 22, 12, 12, 22, 23, 23, 23, 23, 23, 23] 
    YD = [19.95, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 22.0, 18.19, 12.0, 26.0, 26.0, 17.79, 14.0, 22.0, 12.0, 12.0, 22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0] 
#     YS = [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 12, 26, 26, 22, 14, 22, 12, 12, 22, 23, 23, 23, 23, 23, 23] 
    GF = [23, 23, 25, 23, 23, 25, 23, 23, 23, 23, 23, 23, 23, 23, 22, 23, 23, 28, 28, 23, 23, 23, 23, 23, 22, 23, 22, 13, 22, 23, 22] 
#     com_price = pd.DataFrame({'province': province, 'BJ': BJ, 'OS': OS, 'YD':YD, 'YS' : YS, 'GF': GF})
    com_price = pd.DataFrame({'province': province, 'BJ': BJ,  'YD':YD, 'GF': GF})
    com_price = com_price.set_index('province')
    return com_price

def get_tm_price():
    province = ['北京', '河北', '吉林', '辽宁', '天津', '黑龙江', '内蒙古', '甘肃', '宁夏', '山西', '青海', '四川', '重庆', '陕西', '江西', '山东', '上海', '西藏', '新疆', '河南', '安徽', '湖北', '江苏', '浙江', '福建', '云南', '广西', '广东', '海南', '贵州', '湖南'] 
    BJ = [10.78, 14.0, 23.0, 22.0, 15.0, 23.0, 23.0, 23.0, 23.0, 22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 19.79, 23.0, 26.0, 26.0, 17.79, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0] 
#     OS = [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 12, 26, 20, 22, 14, 22, 12, 12, 22, 23, 23, 23, 23, 23, 23] 
    YD = [19.95, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 22.0, 18.19, 12.0, 26.0, 20.0, 17.79, 14.0, 22.0, 12.0, 12.0, 22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0]  
#     YS = [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 12, 26, 26, 22, 14, 22, 12, 12, 22, 23, 23, 23, 23, 23, 23] 
    GF = [23, 23, 25, 23, 23, 25, 23, 23, 23, 23, 23, 23, 23, 23, 22, 23, 23, 28, 28, 23, 23, 23, 23, 23, 22, 23, 22, 13, 22, 23, 22]  
#     com_price = pd.DataFrame({'province': province, 'BJ': BJ, 'OS': OS, 'YD':YD, 'YS' : YS, 'GF': GF})
    tm_price = pd.DataFrame({'province': province, 'BJ': BJ,  'YD':YD, 'GF': GF})
    tm_price = tm_price.set_index('province')
    return tm_price

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

def truncate(province):
    '''
    parse citymapping的代号
    '''
    if province not in['黑龙江省', '内蒙古自治区']:
        province = province[:2]
    else:
        province = province[:3]
    return province
def get_city2hub(province2hub):
    city2hub = {}
    for city_id in city_mapping.index.values:
        city2hub[city_id] = province2hub[truncate(city_mapping.loc[city_id, 'Province'])]
    return city2hub
def get_city2hub_price(province2hub, _price):
    city2hub = {}
    for city_id in city_mapping.index.values:
        
        prov = truncate(city_mapping.loc[city_id, 'Province'])
        hub_lst = province2hub[prov]
        city2hub[city_id] = [_price.loc[prov, hub] for hub in hub_lst]
    return city2hub

def get_hubcdf(onecdf, com_city2hub):
    '''
    com数据
    从城市（onecdf）到仓库级别
    输出值[index:sku, columns:hubs(BJ,YD,GF), values:sales]
    '''
    onecdf = onecdf.transpose().reset_index().copy()
    onecdf['hub'] = onecdf['index'].apply(lambda x: com_city2hub[int(x)][0])
    onecdf= onecdf.set_index('index')
    BJ = onecdf[onecdf.hub == 'BJ'].drop(['hub'], axis = 1).sum().transpose()
    YD = onecdf[onecdf.hub == 'YD'].drop(['hub'], axis = 1).sum().transpose()
    GF = onecdf[onecdf.hub == 'GF'].drop(['hub'], axis = 1).sum().transpose()
    hubcdf = pd.concat([BJ,YD,GF], axis = 1)
    hubcdf.columns = ['BJ', 'YD', 'GF']
    for column in hubcdf.columns.values:
        hubcdf[column] = hubcdf[column].astype(float)
    com_df = df[df.CHANNEL == 'com'].copy()
    return hubcdf[hubcdf.index.isin(com_df.sku)]


def get_hubtdf(onetdf, tm_city2hub):
    '''
    tmall数据
    从城市（onecdf）到仓库级别
    输出值[index:sku, columns:hubs(BJ,YD,GF), values:sales]
    '''
    onetdf = onetdf.transpose().reset_index().copy()
    onetdf['hub'] = onetdf['index'].apply(lambda x: tm_city2hub[int(x)][0])
    onetdf= onetdf.set_index('index')
    BJ = onetdf[onetdf.hub == 'BJ'].drop(['hub'], axis = 1).sum().transpose()
    YD = onetdf[onetdf.hub == 'YD'].drop(['hub'], axis = 1).sum().transpose()
    GF = onetdf[onetdf.hub == 'GF'].drop(['hub'], axis = 1).sum().transpose()
    hubtdf = pd.concat([BJ,YD,GF], axis = 1)
    hubtdf.columns = ['BJ', 'YD', 'GF']
    for column in hubtdf.columns.values:
        hubtdf[column] = hubtdf[column].astype(float)
    tm_df = df[df.CHANNEL == 'tmall'].copy()
    return hubtdf[hubtdf.index.isin(tm_df.sku)]

def yunlong_cross(pred_df,truth_df):
    '''

    :param df: a DataFrame,columns like this [sku,BJ,YD,GF]

    :return:
    '''
    for idx in pred_df.index:
        scale = truth_df.loc[idx,['BJ','YD','GF']].sum()
        ratio = pred_df.loc[idx,['BJ','YD','GF']]/pred_df.loc[idx,['BJ','YD','GF']].sum()
        pred_df.loc[idx,'BJ'],pred_df.loc[idx,'YD'],pred_df.loc[idx,'GF'] = \
            [ratio[i]*scale for i in range(3)]
    df = pred_df.loc[:,['BJ','YD','GF']]-truth_df.loc[:,['BJ','YD','GF']]
#     print(df.abs().sum().sum()/2)
    return (df.abs().sum().sum()/2)

def get_hubpcdf(df): 
    '''
    com数据
    将预测值整理成输出值：[index:sku, columns:hubs(BJ,YD,GF), values:sales]
    '''

    com_df = df[df.CHANNEL == 'com'].copy()
    hubcdf = get_hubcdf(df=df)
    total_qty = hubcdf.sum(axis = 1).reset_index()
    total_qty.columns = ['sku', 'label']
    BJ = com_df[com_df.DIGITAL_HUB == 'BJ'].copy()[['sku', 'qty']].rename(columns = {'qty': 'BJ'}).set_index('sku')
    YD = com_df[com_df.DIGITAL_HUB == 'YD'].copy()[['sku', 'qty']].rename(columns = {'qty': 'YD'}).set_index('sku')
    GF = com_df[com_df.DIGITAL_HUB == 'GF'].copy()[['sku', 'qty']].rename(columns = {'qty': 'GF'}).set_index('sku')
    com_h = pd.concat([BJ,YD,GF], axis = 1)
    com_h.fillna(0, inplace = True)
    com_ratio = com_h.copy()
    
    com_ratio['total'] = com_ratio.sum(axis = 1)

    com_ratio['BJ'] = com_ratio['BJ'] / com_ratio['total']
    com_ratio['YD'] = com_ratio['YD'] / com_ratio['total']
    com_ratio['GF'] = com_ratio['GF'] / com_ratio['total']
    
    com_ratio = com_ratio.reset_index()
    com_ratio = com_ratio.rename(columns = {'index':'sku'})

    com_ratio = com_ratio.merge(total_qty, on = 'sku',how = 'right')
    com_ratio.fillna(float(1/3),inplace = True)
#     print('com:', com_ratio.isnull().mean())
    hubpcdf = com_ratio.copy()
    hubpcdf['BJ'] = com_ratio['BJ'] * com_ratio['label']
    hubpcdf['YD'] = com_ratio['YD'] * com_ratio['label']
    hubpcdf['GF'] = com_ratio['GF'] * com_ratio['label']
    hubpcdf = hubpcdf.set_index('sku')[['BJ', 'GF', 'YD']]
    hubpcdf.fillna(0,inplace = True)
    return hubpcdf

def get_hubptdf(df): 
    '''
    tmall数据
    将预测值整理成输出值：[index:sku, columns:hubs(BJ,YD,GF), values:sales]
    '''
    tm_df = df[df.CHANNEL == 'tmall'].copy()
    hubtdf = get_hubtdf(df = df)
    total_qty = hubtdf.sum(axis = 1).reset_index()
    total_qty.columns = ['sku', 'label']
    BJ = tm_df[tm_df.DIGITAL_HUB == 'BJ'].copy()[['sku', 'qty']].rename(columns = {'qty': 'BJ'}).set_index('sku')
    YD = tm_df[tm_df.DIGITAL_HUB == 'YD'].copy()[['sku', 'qty']].rename(columns = {'qty': 'YD'}).set_index('sku')
    GF = tm_df[tm_df.DIGITAL_HUB == 'GF'].copy()[['sku', 'qty']].rename(columns = {'qty': 'GF'}).set_index('sku')
    tm_h = pd.concat([BJ,YD,GF], axis = 1)
    tm_h.fillna(0, inplace = True)
    tm_ratio = tm_h.copy()
    
    tm_ratio['total'] = tm_ratio.sum(axis = 1)
#     print('tm1 \n:',tm_ratio.isnull().mean())
    tm_ratio['BJ'] = tm_ratio.apply(lambda row: row['BJ']/row['total'], axis = 1)
    
    tm_ratio['YD'] = tm_ratio.apply(lambda row: row['YD']/row['total'], axis = 1)
    tm_ratio['GF'] = tm_ratio.apply(lambda row: row['GF']/row['total'], axis = 1)
    
    tm_ratio = tm_ratio.reset_index()
    tm_ratio = tm_ratio.rename(columns = {'index':'sku'})
    tm_ratio = tm_ratio.merge(total_qty, on = 'sku',how = 'right')
#     print('tm: \n',tm_ratio.isnull().mean())
    tm_ratio.fillna(float(1/3),inplace = True)
    
    hubptdf = tm_ratio.copy()
    hubptdf['BJ'] = tm_ratio['BJ'] * tm_ratio['label']
    hubptdf['YD'] = tm_ratio['YD'] * tm_ratio['label']
    hubptdf['GF'] = tm_ratio['GF'] * tm_ratio['label']
    hubptdf = hubptdf.set_index('sku')[['BJ', 'GF', 'YD']]
    hubptdf.fillna(0,inplace = True)
    return hubptdf


def get_tbench(df ):
    '''
    tmall benchmark, 格式同上
    '''
    tm_df = df[df.CHANNEL == 'tmall'].copy()
    hubtdf = get_hubtdf(df = df)
    total_qty = hubtdf.sum(axis = 1).reset_index()
    total_qty.columns = ['sku', 'label']
    BJ = tm_df[tm_df.DIGITAL_HUB == 'BJ'].copy()[['sku', 'benchmark']].rename(columns = {'benchmark': 'BJ'}).set_index('sku')
    YD = tm_df[tm_df.DIGITAL_HUB == 'YD'].copy()[['sku', 'benchmark']].rename(columns = {'benchmark': 'YD'}).set_index('sku')
    GF = tm_df[tm_df.DIGITAL_HUB == 'GF'].copy()[['sku', 'benchmark']].rename(columns = {'benchmark': 'GF'}).set_index('sku')
    tm_h = pd.concat([BJ,YD,GF], axis = 1)
    tm_h.fillna(0, inplace = True)
    tm_ratio = tm_h.copy()
    tm_ratio['total'] = tm_ratio.sum(axis = 1)

    tm_ratio['BJ'] = tm_ratio['BJ'] / tm_ratio['total']
    tm_ratio['YD'] = tm_ratio['YD'] / tm_ratio['total']
    tm_ratio['GF'] = tm_ratio['GF'] / tm_ratio['total']

    tm_ratio = tm_ratio.reset_index()
    tm_ratio = tm_ratio.rename(columns = {'index':'sku'})
    tm_ratio = tm_ratio.merge(total_qty, on = 'sku',how = 'right')
#     print('tbench:',tm_ratio.isnull().mean())
    tm_ratio.fillna(float(1/3),inplace = True)
    
    hubptdf = tm_ratio.copy()
    hubptdf['BJ'] = tm_ratio['BJ'] * tm_ratio['label']
    hubptdf['YD'] = tm_ratio['YD'] * tm_ratio['label']
    hubptdf['GF'] = tm_ratio['GF'] * tm_ratio['label']
    hubptdf = hubptdf.set_index('sku')[['BJ', 'GF', 'YD']]
    hubptdf.fillna(0,inplace = True)
    return hubptdf

def get_cbench( df ):
    '''
    com benchmark, 格式同上
    '''
    com_df = df[df.CHANNEL == 'com'].copy()
    hubcdf = get_hubcdf(df=df)
    total_qty = hubcdf.sum(axis = 1).reset_index()
    total_qty.columns = ['sku', 'label']
    BJ = com_df[com_df.DIGITAL_HUB == 'BJ'].copy()[['sku', 'benchmark']].rename(columns = {'benchmark': 'BJ'}).set_index('sku')
    YD = com_df[com_df.DIGITAL_HUB == 'YD'].copy()[['sku', 'benchmark']].rename(columns = {'benchmark': 'YD'}).set_index('sku')
    GF = com_df[com_df.DIGITAL_HUB == 'GF'].copy()[['sku', 'benchmark']].rename(columns = {'benchmark': 'GF'}).set_index('sku')
    com_h = pd.concat([BJ,YD,GF], axis = 1)
    com_h.fillna(0, inplace = True)
    com_ratio = com_h.copy()
    
    com_ratio['total'] = com_ratio.sum(axis = 1)
    com_ratio['BJ'] = com_ratio['BJ'] / com_ratio['total']
    com_ratio['YD'] = com_ratio['YD'] / com_ratio['total']
    com_ratio['GF'] = com_ratio['GF'] / com_ratio['total']
    
    com_ratio = com_ratio.reset_index()
    com_ratio = com_ratio.rename(columns = {'index':'sku'})

    com_ratio = com_ratio.merge(total_qty, on = 'sku',how = 'right')
#     print('com bench:', com_ratio.isnull().mean())
    com_ratio.fillna(float(1/3),inplace = True)
    
    hubpcdf = com_ratio.copy()
    hubpcdf['BJ'] = com_ratio['BJ'] * com_ratio['label']
    hubpcdf['YD'] = com_ratio['YD'] * com_ratio['label']
    hubpcdf['GF'] = com_ratio['GF'] * com_ratio['label']
    hubpcdf = hubpcdf.set_index('sku')[['BJ', 'GF', 'YD']]
    hubpcdf.fillna(0,inplace = True)
    return hubpcdf


def compute_cross_region_cost(onedf, hubpdf, platform, i = 0, cost = 0):
    '''
    计算函数，使用DP，以城市pref仓库的优先级为顺序，依次满足从高到低的优先级的需求
    '''

    onedf = onedf
    hubpdf = hubpdf
    platform = platform
    cost_i = cost
    
    city_mapping = pd.read_csv('/backup/POC_Data/cityMapping.csv')
    city_mapping = city_mapping.reset_index()

    com_pref = get_com_pref()
    tm_pref = get_tmall_pref()
    com_price = get_com_price()
    tm_price = get_tm_price()
    tm_province2hub = get_pref_dict(tm_pref)
    com_province2hub = get_pref_dict(com_pref)
    tm_city2hub = get_city2hub(tm_province2hub)
    com_city2hub = get_city2hub(com_province2hub)
    tm_city2price = get_city2hub_price(tm_province2hub, tm_price)
    com_city2price = get_city2hub_price(com_province2hub, com_price)
    
    normal_pref = get_normal_pref()
    normal_price = get_normal_price(normal_pref)
    normal_province2hub = get_pref_dict(normal_pref)
    normal_city2hub = get_city2hub(normal_province2hub)
    normal_city2price = get_city2hub_price(normal_province2hub, normal_price)
    if platform == 'com':
        price_dic = com_city2price
        city2hub = com_city2hub
    elif platform == 'tmall':
        price_dic = tm_city2price
        city2hub = tm_city2hub
    elif platform == 'normal':
        price_dic = normal_city2price
        city2hub = normal_city2hub
        
#     for city in tqdm(onedf):
    for city in onedf:
        hub = city2hub[int(city)][i]
        hub_info = hubpdf[hub]
        city_info = onedf[city]
        hub_info = hubpdf[hub]
        for sku in city_info:
            
            m = hub_info.get(sku, 0)
            n = city_info.get(sku, 0)
            if  n <= m:
                cost_i += n*price_dic[int(city)][i]
                m = m - n
                n = n - n
                hub_info[sku] = m
                city_info[sku] = n

            else:
                cost_i += m*price_dic[int(city)][i]         
                n = n - m
                m = m - m
                hub_info[sku] = m
                city_info[sku] = n

        
    i += 1
#     print(cost_i)
    if i != 3:
        
        return compute_cross_region_cost(onedf = onedf, hubpdf = hubpdf, platform = platform, i = i, cost = cost_i)
    else:
        return cost_i, city_info

def main():
    tmcosts = []
    comcosts = []
    cbcosts = []
    tbcosts = []
    tmtotal = []
    comtotal = []
    tmcross_qty = []
    cmcross_qty = []
    base_tmcross_qty = []
    base_comcross_qty = []
    onecdf_origin = get_onecdf()
    onetdf_origin = get_onetdf()
    for i in tqdm(range(1, 14)):
        onecdf = onecdf_origin.copy()
        onetdf = onetdf_origin.copy()
        df = pd.read_csv('./prediction/hub_prediction_11.5_11.18_sum/' + str(i) +'w_ahead_v2.csv')
        hubcdf = get_hubcdf(df = df, com_city2hub = normal_city2hub)
        hubtdf = get_hubtdf(df = df, tm_city2hub = normal_city2hub)
        com_df = df[df.CHANNEL == 'com'].copy()
        tm_df = df[df.CHANNEL == 'tmall'].copy()
        hubpcdf = get_hubpcdf(df)
        comtotal.append(hubpcdf.sum().sum())
        hubptdf = get_hubptdf(df)
        tmtotal.append(hubptdf.sum().sum())
        onetdf_dict1 = onetdf.copy().to_dict()
        onetdf_dict2 = onetdf.copy().to_dict()
        onecdf_dict1 = onecdf.copy().to_dict()
        onecdf_dict2 = onecdf.copy().to_dict()
        hubpcdf_dict = hubpcdf.to_dict()
        hubptdf_dict = hubptdf.to_dict()
        cbench = get_cbench(df = com_df)
        tbench = get_tbench(df = tm_df)
        cbench_dict = cbench.to_dict()
        tbench_dict = tbench.to_dict()
        com_cost, _ = compute_cross_region_cost(onedf = onecdf_dict1, hubpdf = hubpcdf_dict, platform = 'com', i = 0, cost = 0)
        comcosts.append(np.round(com_cost,2))
        cbcost, _ = compute_cross_region_cost(onedf = onecdf_dict2, hubpdf = cbench_dict, platform = 'com', i = 0, cost = 0)
        cbcosts.append(np.round(cbcost,2))
        tm_cost, _ = compute_cross_region_cost(onedf = onetdf_dict1, hubpdf = hubptdf_dict, platform = 'tmall', i = 0, cost = 0)
        tmcosts.append(np.round(tm_cost,2))
        tbcost, _ = compute_cross_region_cost(onedf = onetdf_dict2, hubpdf = tbench_dict, platform = 'tmall', i = 0, cost = 0)
        tbcosts.append(np.round(tbcost,2))
        tmcross_qty.append(yunlong_cross(tbench.reset_index(), hubtdf.reset_index() ))
        base_tmcross_qty.append(yunlong_cross(hubptdf.reset_index(), hubtdf.reset_index() ))
        cmcross_qty.append(yunlong_cross(cbench.reset_index(), hubcdf.reset_index() ))
        base_comcross_qty.append(yunlong_cross(hubpcdf.reset_index(), hubcdf.reset_index() ))
        result = pd.DataFrame({'tmall_cost': tmcosts,'tmall_benchmark':tbcosts, 'tmall_total_qty': tmtotal,
            'com_cost': comcosts, 'com_benchmark':cbcosts, 'com_total_qty': comtotal，
            'tmall_cross_qty':tmcross_qty, 'com_cross_qty':cmcross_qty, 'base_tmall_cross_qty': base_tmcross_qty, 'base_com_cross_qty':base_comcross_qty
            })
        return result