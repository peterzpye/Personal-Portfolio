import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
import os
from tqdm import tqdm
import shutil
import hashlib

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


available_time = 13
predict = True
com_city_dir_path = '/backup/ND/label_files/city_level/com_file_zhipeng/'
tmall_city_dir_path = '/backup/ND/label_files/city_level/tmall_file_zhipeng/'
output_dir_path = '/backup/ND/feature_files/city_level/feature_eng_yzp/'




pdp_path = '/backup/ND/label_files/city_level/pdp_sum_zhipeng.csv'
traffic_path = '/backup/ND/feature_files/traffic_feature_yzp.csv'
'''
-----------------------------------------------------Utility Func-----------------------------------------------------------------------
'''


def sha256_enc(hash_string):
    sha_signature = hashlib.sha256(hash_string.encode()).hexdigest()
    return sha_signature

def read_sku_list():
    dir_path = '/home/yezhipeng/sku_normal_longtail_split/'
    with open(dir_path + 'longtail_sku.txt', 'r') as file:
        longtail_sku = file.read().split('\n')
    with open(dir_path + 'normal_sku.txt', 'r') as file:
        normal_sku = file.read().split('\n')
    with open(dir_path + 'seasonal_sku.txt', 'r') as file:
        seasonal_sku = file.read().split('\n')
    return normal_sku, longtail_sku, seasonal_sku

def get_product_df():
    import pyarrow.orc as orc
    target_path = '/backup/ND/GC Product Master/part-00000-efc19fcb-f673-4438-af65-02560d10d5ac-c000.snappy.orc'
    with open(target_path, 'rb') as file:
        data = orc.ORCFile(file)
        product = data.read().to_pandas()
    if 'product_info.csv' not in os.listdir('/home/yezhipeng/info/'):
        product.to_csv('/home/yezhipeng/info/product_info.csv')
        print('copy saved to /home/yezhipeng/info/')
    return product

def get_sku_list():
    target_path = '/home/yh/11/info/sku_list.csv'
    df = pd.read_csv(target_path)
    if 'sku_list.csv' not in os.listdir('/home/yezhipeng/info/'):
        df.to_csv('/home/yezhipeng/info/sku_list.csv')
        print('copy saved to /home/yezhipeng/info/')
    return df['Prod Cd'].tolist()

def add_pred_sku(df):
    sku_list = get_sku_list()
    new_date_dfs = [df]
    for sku in sku_list:
        df_sku = df.loc[df.sku == sku]
        if len(df_sku) != 0:
            max_date = pd.to_datetime(df_sku.date.max()).strftime('%Y-%m-%d')
            if max_date <= '2020-11-18':
                new_date = pd.date_range(start = '2020-11-05', end = '2020-11-18', freq = 'D').strftime('%Y-%m-%d')
                new_date_df = pd.DataFrame({'date':new_date, 'sku': sku, 'qty':np.nan})
                new_date_dfs.append(new_date_df)
        else:
            new_date = pd.date_range(start = '2020-11-05', end = '2020-11-18', freq = 'D').strftime('%Y-%m-%d')
            new_date_df = pd.DataFrame({'date':new_date, 'sku': sku, 'qty':np.nan})
            new_date_dfs.append(new_date_df)
    return pd.concat(new_date_dfs)

def get_do11():
    dates = []
    for year in range(2012, 2021):
        for day in range(5, 19):
            if len(str(day)) == 1:
                dates.append(f'{year}-11-0{day}')
            else:
                dates.append(f'{year}-11-{day}')
    return dates


'''
-----------------------------Feature Func-----------------------------
'''

def complete_holiday_df(HOLIDAY_REF):
    for dt in [20140618, 20140612,20140613,20140614, 20140615, 20140616,20140619, 20140620, 20140621,
               20150618, 20150612,20150613,20150614, 20150615, 20150616,20150619, 20150620, 20150621,
               20160618, 20160612,20160613,20160614, 20160615, 20160616,20160619, 20160620, 20160621,
               20170618, 20170612,20170613,20170614, 20170615, 20170616,20170619, 20170620, 20170621,
               20180618, 20180612,20180613,20180614, 20180615, 20180616,20180619, 20180620, 20180621,
               20190618, 20190612,20190613,20190614, 20190615, 20190616,20190619, 20190620, 20190621,
               20200618, 20200612,20200613,20200614, 20200615, 20200616,20200619, 20200620, 20200621]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = '6.18'
    for dt in [20141111, 20141105, 20141106, 20141107, 20141108, 20141109, 20141110, 20141112, 20141113, 20141114, 20141115, 20141116, 20141117, 20141118,
               20151111, 20151105, 20151106, 20151107, 20151108, 20151109, 20151110, 20151112, 20151113, 20151114, 20151115, 20151116, 20151117, 20151118,
               20161111, 20161105, 20161106, 20161107, 20161108, 20161109, 20161110, 20161112, 20161113, 20161114, 20161115, 20161116, 20161117, 20161118,
               20171111, 20171105, 20171106, 20171107, 20171108, 20171109, 20171110, 20171112, 20171113, 20171114, 20171115, 20171116, 20171117, 20171118,
               20181111, 20181105, 20181106, 20181107, 20181108, 20181109, 20181110, 20181112, 20181113, 20181114, 20181115, 20181116, 20181117, 20181118,
               20191111, 20191105, 20191106, 20191107, 20191108, 20191109, 20191110, 20191112, 20191113, 20191114, 20191115, 20191116, 20191117, 20191118,
               20201111, 20201105, 20201106, 20201107, 20201108, 20201109, 20201110, 20201112, 20201113, 20201114, 20201115, 20201116, 20201117, 20201118]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'Do-11'
    for dt in [20140101, 20150101, 20150102, 20150103, 20160101, 20160102, 20160103, 20161231, 
               20170101, 20170102, 20171230, 20171231, 20180101, 20181230, 20181231, 20190101,
               20200101]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'new_year'
    for dt in [20140131, 20140201, 20140202, 20140203, 20140204, 20140205, 20140206, 20150218, 
               20150219, 20150220, 20150221, 20150222, 20150223, 20150224, 20160207, 20160208, 
               20160209, 20160210, 20160211, 20160212, 20160213, 20170127, 20170128, 20170129, 
               20170130, 20170131, 20170201, 20170202, 20180215, 20180216, 20180217, 20180218, 
               20180219, 20180220, 20180221, 20190204, 20190205, 20190206, 20190207, 20190208, 
               20190209, 20190210, 20200124, 20200125, 20200126, 20200127, 20200128, 20200129,
               20200130, 20200131]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'spring'
    for dt in [20140214, 20150214, 20160214, 20170214, 20180214, 20190214, 20200214]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'valentine'
    for dt in [20140405, 20140407, 20150405, 20150406, 20160402, 20160403, 20160404, 20170402, 
               20170403, 20170404, 20180405, 20180406, 20180407, 20190405, 20190406, 20190407, 
               20200404, 20200405, 20200406]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'qingming'
    for dt in [20140501, 20140502, 20140503, 20150501, 20150502, 20150503,
               20160430, 20160501, 20160502, 20170429, 20170430, 20170501,
               20180429, 20180430, 20180501, 20190429, 20190430, 20190501,
               20200501, 20200502, 20200503]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'labor'
    for dt in [20140531, 20140601, 20140602, 20150620, 20150622, 20160609, 20160610, 20160611, 
               20170529, 20170530, 20170528, 20180616, 20180617, 20180618, 20190607, 20190608, 
               20190609, 20200625, 20200626, 20200627]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'dragon_boat'
    for dt in [20140906, 20140907, 20140908, 20150927, 20160915, 20160916, 20160917, 
               20180923, 20180924, 20180922, 20190913, 20190914, 20190915, 20201001]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'moon_cake'
    for dt in [20141001, 20141002, 20141003, 20141004, 20141005, 20141006, 20141007, 
               20151001, 20151002, 20151003, 20151004, 20151005, 20151006, 20151007, 
               20161001, 20161002, 20161003, 20161004, 20161005, 20161006, 20161007, 
               20171001, 20171002, 20171003, 20171008, 20171004, 20171005, 20171006, 
               20171007, 20181001, 20181002, 20181003, 20181004, 20181005, 20181006, 
               20181007, 20191001, 20191002, 20191003, 20191004, 20191005, 20191006, 
               20191007, 20201001, 20201002, 20201003, 20201004, 20201005, 20201006,
               20201007, 20201008]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'national'
#     串休的日期变为工作日
    for dt in [20140126, 20140208, 20140504, 20140928, 20141011, 20150104, 20150215, 20150228, 
               20151010, 20160206, 20160214, 20160612, 20160918, 20161008, 20161009, 20170122, 
               20170204, 20170401, 20170527, 20170930, 20180211, 20180224, 20180408, 20180428,
               20180929, 20180930, 20190929, 20191012]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = 'normal'
    return HOLIDAY_REF



def add_holiday(holiday_feature_df, holiday_df):

    fres_list = []
    bres_list = []
    for x in holiday_feature_df.date:
        fstart_date = x
        fend_date = (pd.to_datetime(fstart_date) + pd.to_timedelta(3, unit='D')).strftime('%Y-%m-%d')
        bend_date = x
        bstart_date = (pd.to_datetime(bend_date) - pd.to_timedelta(3, unit='D')).strftime('%Y-%m-%d')
        fres = len(holiday_df.query(f'"{fstart_date}" <= date < "{fend_date}" and day_desc != "normal"'))
        bres = len(holiday_df.query(f'"{bstart_date}" <= date < "{bend_date}" and day_desc != "normal"'))
        fres_list.append(fres)
        bres_list.append(bres)
    holiday_feature_df['holiday_after'] = fres_list
    holiday_feature_df['holiday_before'] = bres_list
    if 'day_desc'  in (holiday_feature_df.columns.values):
        holiday_feature_df = holiday_feature_df.drop('day_desc', axis = 1)
    
    return holiday_feature_df

def traffic_features(df):
    channel_tojoin_mean = df.groupby(['date','sku']).sum().shift(available_time*7).rolling(7, min_periods = 1).mean()[['daily_traffic', 'daily_buyer_count']].copy()
    channel_tojoin_mean.columns = ['daily_traffic_1wmean', 'daily_buyer_count_1wmean']
    channel_tojoin_std = df.groupby(['date','sku']).sum().shift(available_time*7).rolling(7, min_periods = 1).std()[['daily_traffic', 'daily_buyer_count']].copy()
    channel_tojoin_std.columns = ['daily_traffic_1wstd', 'daily_buyer_count_1wstd']
    channel_tojoin = pd.concat([channel_tojoin_mean, channel_tojoin_std], axis = 1).reset_index()
#     return df
    df = df.merge(channel_tojoin, on = ['date', 'sku'], how = 'left')
    return df

def  whole_city_metrics(df):
    whole_city = df.groupby('date').agg({'qty':'sum'})
    whole_city = whole_city.rename(columns = {'qty':'city_day_total'}).reset_index()
    
    df = df.merge(whole_city, on = 'date', how = 'left')
    return df

def last_year_do11_sales(df):
    last_year_sales = df.groupby('year').agg({'qty':'sum'}).rename(columns = {'qty':'last_year_do11_sum'}).reset_index()
    df.merge(last_year_sales, on = 'year', how = 'left')
    return df


def day_features(df_short):
    df_short.date = pd.to_datetime(df_short.date)
    df_short['90d_ago'] = df_short.groupby('sku')['qty'].shift(90)
    df_short['365d_ago'] = df_short.groupby('sku')['qty'].shift(365)
    df_short['90d_ago_1w_mean'] = df_short.groupby('sku')['qty'].shift(90).rolling(7, min_periods = 1).mean()
    df_short['90d_ago_4w_mean'] = df_short.groupby('sku')['qty'].shift(90).rolling(7*4, min_periods = 1).mean()
    df_short['365d_ago_1w_mean'] = df_short.groupby('sku')['qty'].shift(365).rolling(7, min_periods = 1).mean()
    df_short['90d_ago_1w_std'] = df_short.groupby('sku')['qty'].shift(30).rolling(7, min_periods = 1).std()
    df_short['365d_ago_1w_std'] = df_short.groupby('sku')['qty'].shift(365).rolling(7, min_periods = 1).std()
    df_short['1w_ago_1w_sum_qty'] = df_short.groupby('sku')[['date', 'qty']].shift(7).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['2w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*2).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['3w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*3).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['4w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*4).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['5w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*5).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['6w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*6).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['7w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*7).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['8w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*8).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['9w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*9).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['10w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*10).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['11w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*11).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['12w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*12).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    df_short['13w_ago_1w_sum_qty'] = df_short.groupby('sku')['date','qty'].shift(7*13).rolling(7, min_periods = 1, on = 'date').sum()['qty']
    # all these variables are meant to be day mean of past 4 weeks
    df_short['1w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['2w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*2).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['3w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*3).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['4w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*4).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['5w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*5).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['6w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*6).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['7w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*7).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['8w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*8).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['9w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*9).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['10w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*10).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['11w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*11).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['12w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*12).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']
    df_short['13w_ago_4w_sum_qty'] = (df_short.groupby('sku')['date','qty'].shift(7*13).rolling(7*4, min_periods = 1, on = 'date').sum()/28)['qty']

    # whole city metrics
    df_short['1w_ago_1w_sum_city'] = df_short.groupby('sku')[['date','city_day_total']].shift(7).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['2w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*2).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['3w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*3).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['4w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*4).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['5w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*5).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['6w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*6).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['7w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*7).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['8w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*8).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['9w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*9).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['10w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*10).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['11w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*11).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['12w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*12).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']
    df_short['13w_ago_1w_sum_city'] = df_short.groupby('sku')['date','city_day_total'].shift(7*13).rolling(7, min_periods = 1, on = 'date').sum()['city_day_total']


    df_short.date = df_short.date.dt.strftime('%Y-%m-%d')
    return df_short

'''
-----------------------------------------------------Assembling Data Func--------------------------------------------
'''

def assemble_one_data(label_path, pdp_path, traffic_path, dtc_path):
    '''
    assemble the dataset of one city
    modified a few features
    '''
#     label = pd.read_csv(label_path)
#     label = label.drop(['level_1_x', 'level_1_y'], axis = 1)
    label = pd.read_csv(label_path, usecols = ['date', 'sku', 'qty', 'day_desc', 'holiday_after','holiday_before'])
    
    # elongate data to include 2020-11-18

    label = add_pred_sku(label)
#     print(len(label))
    #label features
    label = whole_city_metrics(label)
    label = day_features(label)
    in_sku = []


    for col in ['day_desc', 'holiday_after', 'holiday_before']:
        if col in label.columns:
            label = label.drop(col, axis = 1)
    label.date = label.date.apply(lambda x: x[:10])
    pdp = pd.read_csv(pdp_path)
    traffic = pd.read_csv(traffic_path)
    dtc = pd.read_csv(dtc_path)
    dtc = dtc.rename(columns = {'ORDER_DT':'date', 'UNIV_STYLE_COLOR_CD':'sku'})
    dtc.date = dtc.date.apply(lambda x: x[:10])
    df_product = label.merge(pdp, on = 'date', how = 'left')
    
    df_product = df_product.merge(traffic, on = 'date', how = 'left')
    df_product = df_product.merge(dtc, on = ['date', 'sku'], how = 'left')

#     print('before holiday shape: ', df_product.shape)
    filtered_holiday = []
    holiday_df = complete_holiday_df(pd.DataFrame())
    
#     #holiday feature
    df_product = df_product.merge(holiday_df, on = 'date', how = 'left')
    
    holiday_feature_df = df_product[['date']].drop_duplicates().copy()
    
    holiday_feature_df = add_holiday(holiday_feature_df, holiday_df)

    df_product = df_product.merge(holiday_feature_df, on = 'date', how = 'left')
    
    
    
    df_product['date'] = pd.to_datetime(df_product.date)
    df_product['dayofweek'] = df_product.date.dt.dayofweek
    df_product['dayofyear'] = df_product.date.dt.dayofyear
    df_product['year'] = df_product.date.dt.year
    df_product.date = df_product.date.dt.strftime('%Y-%m-%d')

# #     #traffic features
    df_product = traffic_features(df_product)
    
#     print('traffic shape: ', df_product.shape)
    # product features
    product_info = get_product_df()
    sku_count = product_info.groupby('PRODUCT_CODE').count()['ID'].reset_index()
    sku_count.columns = ['sku', 'sku_type_count']
    product_features = product_info.groupby('PRODUCT_CODE').tail(1).reset_index()[['PRODUCT_CODE','KIDS_AGE', 'LAUNCH_TIER','GENDER_CODE','FIT_CODE', 'CATEGORY_CODE']]
    product_features['KIDS_AGE'] = product_features['KIDS_AGE'].apply(lambda x: 'None' if x =='' else x)
    product_features['LAUNCH_TIER'] = product_features['LAUNCH_TIER'].apply(lambda x: 'None' if x =='' else x)
    product_features.columns = ['sku', 'KIDS_AGE','LAUNCH_TIER', 'GENDER_CODE','FIT_CODE', 'CATEGORY_CODE']
    df_product = df_product.merge(sku_count, on = 'sku', how = 'left')
    df_product = df_product.merge(product_features, on = 'sku', how = 'left')
#     print('product shape: ', df_product.shape)
#     #collect holiday dates
    
    
    filtered_holiday = get_do11()
#     filtered_holiday = holiday_df.date.values

    
#     #trim datasets to holiday only
    df_holiday = df_product.loc[df_product.date.isin(filtered_holiday)]
    df_holiday = last_year_do11_sales(df_holiday).set_index('date')

#     print(df_holiday.index.max())
#     print('holiday trim: ', df_holiday.shape)
#     #encode sku name
#     df_holiday.sku = df_holiday.sku.apply(lambda x: sha256_enc(x))

    normal_sku, longtail_sku, seasonal_sku = read_sku_list()

    df_longtail_holiday = df_holiday[df_holiday['sku'].isin(longtail_sku)].copy()
    df_normal_holiday = df_holiday[df_holiday['sku'].isin(normal_sku)].copy()
    df_seasonal_holiday = df_holiday[df_holiday['sku'].isin(seasonal_sku)].copy()

#     print('file split finished!')
    return df_longtail_holiday, df_normal_holiday, df_seasonal_holiday


def get_one_result(city):
    city = str(city)

    if city not in os.listdir('/ND_result_yzp/'):
        if city[-9:] == 'tmall.csv':
            label_path = f'/backup/ND/label_files/city_level/tmall_file_zhipeng/{city}'
        elif city[-7:] == 'com.csv':
            label_path = f'/backup/ND/label_files/city_level/com_file_zhipeng/{city}'
        else:
            raise ValueError(f'invalid city file name input: {city}')

        dtc_path = f'/backup/ND/label_files/city_level/dtc_zhipeng/{city}'

        df_longtail_holiday, df_normal_holiday, df_seasonal_holiday = assemble_one_data(label_path, pdp_path, traffic_path, dtc_path)
        df_longtail_holiday.to_csv(output_dir_path +  'longtail_' + city )
        df_normal_holiday.to_csv(output_dir_path + 'normal_' + city )
        df_seasonal_holiday.to_csv(output_dir_path + 'seasonal_' + city )

def main():
    coms = [name for name in os.listdir(com_city_dir_path)]
    for city in tqdm(coms):
        # print(city)
        get_one_result(city)
    tmalls = [name for name in os.listdir(tmall_city_dir_path)]
    for city in tqdm(tmalls):
        get_one_result(city)