import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn

import os
from zipfile import ZipFile 
from tqdm import tqdm
import shutil


def fill_continum(df):
    df = df.sort_values(by = 'date').copy()
    last = df.iloc[-1:].date
    first = df.iloc[0,:].date
    dates = [first + pd.to_timedelta(i, unit='D') for i in range(df.shape[0])]
    time_df = pd.DataFrame({'date':dates})

    return time_df

def add_rolling_week_sum(df):
    df1 = df.groupby('sku')[['date', 'qty']].rolling(7, on = 'date', min_periods = 1).sum()
    df1 = df1.rename(columns = {'qty':'qty_week'})
    df1 = df1.reset_index()
    return df1

def complete_holiday_df(HOLIDAY_REF):
    for dt in [20170618, 20170612,20170613,20170614, 20170615, 20170616,20170619, 20170620, 20170621,
               20180618, 20180612,20180613,20180614, 20180615, 20180616,20180619, 20180620, 20180621,
               20190618, 20190612,20190613,20190614, 20190615, 20190616,20190619, 20190620, 20190621,
               20200618, 20200612,20200613,20200614, 20200615, 20200616,20200619, 20200620, 20200621]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = '6.18'
    for dt in [20171111, 20171105, 20171106, 20171107, 20171108, 20171109, 20171110, 20171112, 20171113, 20171114, 20171115, 20171116, 20171117, 20171118,
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
        fend_date = (pd.to_datetime(fstart_date) + pd.to_timedelta(7, unit='D')).strftime('%Y-%m-%d')
        bend_date = x
        bstart_date = (pd.to_datetime(bend_date) - pd.to_timedelta(7, unit='D')).strftime('%Y-%m-%d')
        fres = len(holiday_df.query(f'"{fstart_date}" <= date < "{fend_date}" and day_desc != "normal"'))
        bres = len(holiday_df.query(f'"{bstart_date}" <= date < "{bend_date}" and day_desc != "normal"'))
        fres_list.append(fres)
        bres_list.append(bres)
    holiday_feature_df['holiday_after'] = fres_list
    holiday_feature_df['holiday_before'] = bres_list
    return holiday_feature_df

def label_feature_eng(df):
    df_short = df.copy()
    df_short.date = pd.to_datetime(df_short.date)
    dates = df_short.groupby(['sku']).apply(lambda x: fill_continum(x))
    df_short = df_short.merge(dates.reset_index(), on = ['sku', 'date'], how = 'outer' )
    df_short.qty.fillna(0, inplace = True)
    rolling_week_sum = add_rolling_week_sum(df_short)
    
    df_short = df_short.merge(rolling_week_sum, on = ['sku', 'date'], how = 'left')
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
    
    holiday_df = complete_holiday_df(pd.DataFrame())
    df_short = df_short.merge(holiday_df, on = 'date', how = 'left')
    holiday_feature_df = df[['date']].copy()
    holiday_feature_df = add_holiday(holiday_feature_df, holiday_df)
    df_short = df_short.merge(holiday_feature_df, on = 'date', how = 'left')
    
    return df_short

def processor(path, path2, com):
    '''
    convert df with label + holiday feature engineering
    
    path: input file paths
    path2: output path
    '''
    if com == True:
        
        fnames = [name for name in os.listdir(path) if name[-7:] == 'com.csv']
        if len(fnames) == 0:
            raise ValueError(f'check parsing file names{[name[-7:] for name in os.listdir(path)[:10]]}')
        
    elif com == False:
        fnames = [name for name in os.listdir(path) if name[-9:] == 'tmall.csv']
        if len(fnames) == 0:
            raise ValueError(f'check parsing file names{[name[-9:] for name in os.listdir(path)[:10]]}')
        

    for fname in tqdm(fnames):
        if fname not in os.listdir(path2):
            fpath = path + fname 
            df = pd.read_csv(fpath)
            transformed = label_feature_eng(df)
            transformed.to_csv(path2 + fname)
        else:
            continue


def processor_province(path, path2, com):
    '''
    basically the province version of the processor function
    path: input file paths
    path2: output path
    '''
    
    mappath = '/backup/POC_Data/cityMapping.csv'
    citymap = pd.read_csv(mappath)
    city2province = citymap.Province.to_dict()
    prince2city = {}
    for k,v in city2province.items():
        prince2city[v] = []
    for k,v in city2province.items():
        prince2city[v].append(k)
        
        
    if com == True:
        
        fnames = [name for name in os.listdir(path) if name[-7:] == 'com.csv']
        if len(fnames) == 0:
            raise ValueError(f'check parsing file names{[name[-7:] for name in os.listdir(path)[:10]]}')
        
    elif com == False:
        fnames = [name for name in os.listdir(path) if name[-9:] == 'tmall.csv']
        if len(fnames) == 0:
            raise ValueError(f'check parsing file names{[name[-9:] for name in os.listdir(path)[:10]]}')
            
    

    for prince in tqdm(prince2city):
        if com == True:
            cities = [str(city) + '_com.csv' for city in prince2city[prince]]
            prince_name = prince + '_com.csv'
        elif com == False:
            cities = [str(city) + '_tmall.csv' for city in prince2city[prince]]
            prince_name = prince + '_tmall.csv'

        if prince_name not in os.listdir(path2):
            dfs = [pd.read_csv(path + city) for city in cities if city in os.listdir(path)]
            df = pd.concat(dfs, axis = 0)
            df = df.groupby(['date','sku']).sum().reset_index()
            transformed = label_feature_eng(df)
            transformed.to_csv(path2 + prince_name)
        else:
            continue


def main_province():
    path = '/backup/ND/label_files_yh/city_level/'
    try:
        os.mkdir('/backup/ND/label_files/prince_level/com_file_zhipeng')
        print('directory made')
    except:
        print('directory already exist')
        pass
    com_path = '/backup/ND/label_files/prince_level/com_file_zhipeng/'

    processor_province(path, com_path, com = True)
    try:
        os.mkdir('/backup/ND/label_files/prince_level/tmall_file_zhipeng')
        print('directory made')
    except:
        print('directory already exist')
        pass
    tm_path = '/backup/ND/label_files/prince_level/tmall_file_zhipeng/'
    processor_province(path, tm_path, com = False)


def main_city():
    path = '/backup/ND/label_files_yh/city_level/'
    try:
        os.mkdir('/backup/ND/label_files/city_level/com_file_zhipeng')
        print('directory made')
    except:
        print('directory already exist')
        pass
    com_path = '/backup/ND/label_files/city_level/com_file_zhipeng/'

    processor(path, com_path, com = True)
    
    try:
        os.mkdir('/backup/ND/label_files/city_level/tmall_file_zhipeng')
        print('directory made')
    except:
        print('directory already exist')
        pass
    tm_path = '/backup/ND/label_files/city_level/tmall_file_zhipeng/'
    processor(path, tm_path, com = False)