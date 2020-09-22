import numpy as np
import pandas as pd
import seaborn as sb
import os
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 


'''
直接使用main()
'''




import pyarrow.orc as orc
with open('/backup/POC_Data/gc_product_master_v2.orc', 'rb') as file:
    data = orc.ORCFile(file)
    product = data.read().to_pandas()



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
    for dt in [20170618, 20180618, 20190618, 20200618]:
        dt = str(dt)[:4]+'-'+str(dt)[4:6] + '-' + str(dt)[6:]
        idx = len(HOLIDAY_REF)
        HOLIDAY_REF.loc[idx, 'date'] = dt
        HOLIDAY_REF.loc[idx, 'day_desc'] = '6.18'
    for dt in [20171111, 20181111, 20191111, 20201111]:
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
    # 串休的日期变为工作日
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
    holiday_feature_df = holiday_feature_df.drop('day_desc', axis = 1)
    return holiday_feature_df




def feature_engineering(df_short):
    '''
    此函数为第一部分的特征工程，
    输入值为df_short，在main()中的输入值使用的是df，以下内容均以df_short为主
    输出值为加入了各项特征的同样大小的df_short
    '''

#     print('initial_label: ', df_short.qty.sum())
#     print('initial num_sku: ', df_short.sku.nunique())
    #fill zero sales date
    df_short.date = pd.to_datetime(df_short.date)
    dates = df_short.groupby(['sku']).apply(lambda x: fill_continum(x))
    df_short = df_short.merge(dates.reset_index(), on = ['sku', 'date'], how = 'outer' )
    df_short.qty.fillna(0, inplace = True)
#     assert df_short.qty.isnull().any() == False
#     print('fill_zero_sales_date num_sku: ', df_short.sku.nunique())
    # print('fill_zero_sales_date feature label_sum: ',df_short.qty.sum())

    #rolling week sum label
    rolling_week_sum = add_rolling_week_sum(df_short)
    
    df_short = df_short.merge(rolling_week_sum, on = ['sku', 'date'], how = 'left')
#     print('rolling_week feature num_sku: ', df_short.sku.nunique())
#     print('rolling_week feature label_sum: ',df_short.qty.sum())
    #channel
    channel_origin= pd.read_csv('/backup/POC_Data/' + '3.1.3.21.6 tmall_channel_traffic(tmall_mkt_channel).csv')
    channel = channel_origin.rename(columns = {'YEAR_MONTHS_DATE': 'Date'})
    channel_tojoin = pd.DataFrame()
    channel_tojoin['traffic_cly_7d_mean'] = channel.groupby('Date')['TRAFFIC_CLY_CNT'].sum().rolling(7, min_periods = 1).mean()
    channel_tojoin['traffic_cly_7d_std'] = channel.groupby('Date')['TRAFFIC_CLY_CNT'].sum().rolling(7, min_periods = 1).std()
    channel_tojoin['90d_ago_buyers'] = channel.groupby('Date')['DAILY_BUYER_CNT'].sum().shift(90)
    channel_tojoin['90d_ago_buyers_1w_mean'] = channel.groupby('Date')['DAILY_BUYER_CNT'].sum().shift(90).rolling(7).mean()
    channel_tojoin['90d_ago_buyers_1w_std'] = channel.groupby('Date')['DAILY_BUYER_CNT'].sum().shift(90).rolling(7).std()
    channel_tojoin['365d_ago_buyers'] = channel.groupby('Date')['DAILY_BUYER_CNT'].sum().shift(365)
    channel_tojoin['365d_ago_buyers_1w_week'] = channel.groupby('Date')['DAILY_BUYER_CNT'].sum().shift(365).rolling(7).mean()
    channel_tojoin['365d_ago_buyers_1w_std'] = channel.groupby('Date')['DAILY_BUYER_CNT'].sum().shift(365).rolling(7).std()
    channel_tojoin = channel_tojoin.reset_index().rename(columns = {'Date':'date'})
    
    df_short = df_short.merge(channel_tojoin, on = 'date', how = 'left')
#     print('channel feature num_sku: ', df_short.sku.nunique())
    # sales trend
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
    
    #some features from product：product表里面的额外特征
    extra = product[['PRODUCT_CODE', 'CYCLE_YEAR_ABBREVIATION','ATHLETE_NAME','KIDS_AGE', 'LAUNCH_TIER']].rename(columns = {'PRODUCT_CODE':'sku'})
    df_short = df_short.merge(extra, on = ['sku', 'CYCLE_YEAR_ABBREVIATION'], how = 'left')
    
    # holiday feature
    holiday_df = complete_holiday_df(pd.DataFrame())
    holiday_feature_df = df_short[['date', 'day_desc']].drop_duplicates().dropna().copy()
    holiday_feature_df = add_holiday(holiday_feature_df, holiday_df)
    df_short = df_short.merge(holiday_feature_df, on = 'date', how = 'left')
#     print('add holiday feature num_sku: ', df_short.sku.nunique())
    # print('feature_engineering_end_label_sum: ',df_short.qty.sum())
    return df_short


def filter_season_join(df_product, channel):
    '''
    输入值为df_product, main()中使用的为feature_engineering 过后的df
    channel 为tmall 或者 com
    加入额外数据
    按sku类型分为normal，longtail， seasonal
    返还三个处理好的df
    '''



    print('filter_season_join_qty_sum: ', df_product.qty.sum())
    clusters = pd.read_csv('cluster_v3.csv')
    df_product.date = pd.to_datetime(df_product.date).dt.strftime('%Y-%m-%d').astype(str)
    df_product = df_product.merge(clusters.rename(columns = {'PRODUCT_CODE': 'sku'}), 
                                      on = ['sku', 'CYCLE_YEAR_ABBREVIATION'], how = 'left')
    Nike_EBIDTA = {'2018':1807, '2019':2376, '2020': 2490}
    tmall_yoy_gmv = {'2018':2131, '2019':2612, '2020': 3202}
    double11 = {'2018':1682, '2019':2135, '2020': 2684} #数据为last year，eg，2018为2017.11.11数据
    tmall_q_growth = {'2018Q1':np.nan, '2018Q2': np.nan, '2018Q3':0.3, '2018Q4':0.29, '2019Q1': 0.29, 
                      '2019Q2':0.34, '2019Q3':0.26, '2019Q4': 0.24, 
                      '2020Q1':np.nan, '2020Q2':np.nan}
    filtered = df_product
#     print('filtered__qty_sum: ', filtered.qty.sum())
    filtered['Nike_ebidta'] = filtered['date'].apply(lambda x: Nike_EBIDTA.get(x.split('-')[0], np.nan) )
    filtered['tmall_yoy_gmv'] = filtered['date'].apply(lambda x: tmall_yoy_gmv.get(x.split('-')[0], np.nan) )
    filtered['double11'] = filtered['date'].apply(lambda x: double11.get(x.split('-')[0], np.nan) )
    filtered['tmall_q_growth'] = filtered['date'].apply(lambda x: tmall_q_growth.get(x.split('-')[0] +'Q'+ str((int(x.split('-')[1]) +2 )//3), np.nan) )
    filtered.date = filtered.date.astype(str)
    filtered_holiday = []
    holiday_df = complete_holiday_df(pd.DataFrame())
    for x in holiday_df.date.values:
        if x in filtered['date'].values:
            if x not in filtered_holiday:
                filtered_holiday.append(x)
    for x in ['2019-11-01', '2019-11-02','2019-11-03','2019-11-04','2019-11-05','2019-11-06','2019-11-07',
          '2019-11-08','2019-11-09','2019-11-10','2019-11-12','2019-11-13','2019-11-14','2019-11-15','2019-11-16', '2019-11-17','2019-11-18',
          '2018-11-01', '2018-11-02','2018-11-03','2018-11-04','2018-11-05','2018-11-06','2018-11-07',
          '2018-11-08','2018-11-09','2018-11-10','2018-11-12','2018-11-13','2018-11-14','2018-11-15','2018-11-16', '2018-11-17','2018-11-18'
         ]:
        if x not in filtered_holiday:
            filtered_holiday.append(x)
    df_holiday = filtered.copy().set_index('date').loc[filtered_holiday, :]
    if channel == 'com':
        place= '_com.txt'
    else:
        place = '_tm.txt'
    with open('longtail_sku' + place, 'r') as file:
        longtail_sku = file.read().split('\n')
    with open('normal_sku' + place, 'r') as file:
        normal_sku = file.read().split('\n')
    with open('seasonal_sku' + place, 'r') as file:
        seasonal_sku = file.read().split('\n')

#     print('df_holiday_qty_sum: ', df_holiday.qty.sum())
    df_longtail_holiday = df_holiday[df_holiday['sku'].isin(longtail_sku)].copy()
    df_normal_holiday = df_holiday[df_holiday['sku'].isin(normal_sku)].copy()
    df_seasonal_holiday = df_holiday[df_holiday['sku'].isin(seasonal_sku)].copy()
    print('splited_qty_sum: ', df_longtail_holiday.qty.sum() + df_normal_holiday.qty.sum() + df_seasonal_holiday.qty.sum())
    return df_longtail_holiday, df_normal_holiday, df_seasonal_holiday




def MODEL(available_time, target_window, dataset, product):

    available_feature = str(available_time) + 'w_ago_1w_sum_qty'
    base_feature = str(available_time) + 'w_ago_4w_sum_qty'
    num_features = [
        # traffic labels
        # temporary:   
        'traffic_cly_7d_mean', 'traffic_cly_7d_std',  '365d_ago_buyers',
        # product labels
        'DISCOUNT_PCT', 'CFP_INCL_TAX_LC', 'Nike_ebidta','tmall_yoy_gmv','tmall_q_growth',
        # time labels
        'dayofweek', 'dayofyear', 'year','holiday_after', 'holiday_before', 
        #trend labels
        '365d_ago_1w_mean',  base_feature, '90d_ago_1w_mean', '90d_ago_1w_std', 'double11'
                ]
    for i in range(available_time, 14):
        num_features.append(str(i) + 'w_ago_1w_sum_qty')
        
    cat_features = [ 
        #product
        # , 'cluster_price'
        'KIDS_AGE', 'LAUNCH_TIER','GENDER_CODE','FIT_CODE', 'cluster_customer',
        'cluster_popularity', 'sales_volume',
        #time
        'day_desc'

    ]
    if product == True:
        for feature in ['PRODUCT_COLORWAY_CODE', 'CATEGORY_CODE']:
            cat_features.append(feature)
#     print('num: ',num_features)
#     print('cat: ',cat_features)
    def fill_dataset(df_holiday): 
        df3 = df_holiday.copy()

        for column in num_features: 
            df3[column] = df3[column].replace('NA', np.nan)
            df3[column] = df3[column].replace('nan', np.nan)
            try:
                df3[column] = df3[column].astype(float)
            except:
                df3[column] = [str(num).lstrip('[') for num in df3[column]]
                df3[column] = [str(num).rstrip('.]') for num in df3[column]]
                df3[column] = df3[column].astype(float)
            df3[column] = df3[column].replace(np.inf, np.nan)


            if df3[column].isnull().mean() <= 0.5:
                df3[column] = df3[column].fillna(df3[column].mode()[0])
            elif df3[column].isnull().mean() > 0.5 and df3[column].isnull().mean() != 1:
                df3[column] = df3[column].fillna(df3[column].mean())
            else:
                df3[column]=df3[column].fillna(0)
                    
        for column in cat_features:
            df3[column] = df3[column].fillna('NA')
        print(df3[num_features].isnull().any().any())
        print(df3[cat_features].isnull().any().any())
        return df3
#     print('sku num before train test split and encoding: ', dataset.sku.nunique())
    print('label sum before train test split and encoding: ', dataset.qty.sum())
    df_for_training = fill_dataset(dataset)
    df_for_training = df_for_training.reset_index()


    X = df_for_training.iloc[:,4:].copy()
    Y = df_for_training[[target_window]].copy()
    Y.fillna(0, inplace = True)

    cat_dfs = []
    for feature in cat_features:
        dummies = pd.get_dummies(X[feature], prefix=feature)
        cat_dfs.append(dummies)

#     print('num_cat: ', len(cat_dfs))
    X1 = pd.concat(cat_dfs, axis = 1)
    print('categorical features encoding done!')

    from sklearn.preprocessing import MinMaxScaler

    mms = MinMaxScaler()

        
    X[num_features] = mms.fit_transform(X[num_features])
    X2 = X.loc[:, num_features]
#     print('number of numeric features:', len(num_features))
    print('mms done.')

    Xbar = pd.concat([X1, X2], axis = 1)

    from sklearn.model_selection import train_test_split
    
#     test_date = ['2019-11-01', '2019-11-02','2019-11-03','2019-11-04','2019-11-05','2019-11-06','2019-11-07',
#             '2019-11-08','2019-11-09','2019-11-10','2019-11-11','2019-11-12','2019-11-13','2019-11-14','2019-11-15',
#                '2019-11-16', '2019-11-17', '2019-11-18' ]
    test_date = ['2019-11-05','2019-11-06','2019-11-07',
            '2019-11-08','2019-11-09','2019-11-10','2019-11-11','2019-11-12','2019-11-13','2019-11-14','2019-11-15',
               '2019-11-16', '2019-11-17', '2019-11-18' ]

    
#     test_date =  ['2019-11-11']
    df_for_training.date = df_for_training.date.astype(str)
    test_idx = df_for_training.reset_index()[df_for_training.reset_index().date.isin(test_date)].index


    x_test = Xbar.loc[test_idx]
    y_test = Y.loc[test_idx]
    print('y_test:', y_test.sum())
    x = Xbar.drop(test_idx, axis = 0).copy()
    y = Y.drop(test_idx, axis = 0).copy()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder 


    # train val sets
    val_random_state = 42

    x_train, x_val, y_train, y_val = train_test_split(x,y.values, random_state = val_random_state)

    import xgboost as xgb
    params = {
        'learning_rate':0.015,
        'num_leaves':2**10,
        'max_depth' : 7
        
    }
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_val)


    from sklearn.metrics import mean_squared_error, mean_absolute_error

    val_result = []

    # val set
#     print('-------------------------------------------\n')
#     print('val scores')
#     mse = mean_squared_error(y_val, pred)
#     val_result.append(mse)
#     print('mse: ', mse)

#     rmse = np.sqrt(mse)
#     val_result.append(rmse)
#     print('rmse: ', rmse)

#     mae = mean_absolute_error(y_val, pred)
#     val_result.append(mae)
#     print('mae: ', mae)
            

#     smape = SMAPE(y_val, pred)
#     val_result.append(smape)
#     print('smape: ', smape)

#     acc = fullfil_rate(y_val, pred)
#     val_result.append(acc)
#     print('acc: ', acc)

    
    
    # base val
    base_y_val = y_val
    base_pred_val = mms.inverse_transform(x_val[num_features]).T[num_features.index(base_feature)]
    
    base_val_result = []
#     print('-------------------------------------------\n')
#     print('baseline val scores')
#     mse = mean_squared_error(base_y_val, base_pred_val)
#     base_val_result.append(mse)
#     print('mse: ', mse)

#     rmse = np.sqrt(mse)
#     base_val_result.append(rmse)
#     print('rmse: ', rmse)

#     mae = mean_absolute_error(base_y_val, base_pred_val)
#     base_val_result.append(mae)
#     print('mae: ', mae)

#     smape = SMAPE(base_y_val, base_pred_val)
#     base_val_result.append(smape)
#     print('smape: ', smape)

#     acc = accuracy(base_y_val, base_pred_val)
#     base_val_result.append(acc)
#     print('acc: ', acc)
    
    
    
    
    

            # test set
    pred_test = model.predict(x_test)
    test_result = []
#     print('-------------------------------------------\n')
    print('test scores')
    mse = mean_squared_error(y_test, pred_test)
    test_result.append(mse)
    print('mse: ', mse)

#     rmse = np.sqrt(mse)
#     test_result.append(rmse)
#     print('rmse: ', rmse)

#     mae = mean_absolute_error(y_test, pred_test)
#     test_result.append(mae)
#     print('mae: ', mae)

#     smape = SMAPE(y_test, pred_test)
#     test_result.append(smape)
#     print('smape: ', smape)

    acc = accuracy(y_test, pred_test)
    test_result.append(acc)
    print('acc: ', acc)
            
        # base test
    base_y_test = y_test
    base_pred_test = mms.inverse_transform(x_test[num_features]).T[num_features.index(base_feature)]
    base_result = []
#     print('-------------------------------------------\n')
#     print('baseline test scores')
    mse = mean_squared_error(base_y_test, base_pred_test)
#     base_result.append(mse)
    print('mse: ', mse)

#     rmse = np.sqrt(mse)
#     base_result.append(rmse)
#     print('rmse: ', rmse)

#     mae = mean_absolute_error(base_y_test, base_pred_test)
#     base_result.append(mae)
#     print('mae: ', mae)

#     smape = SMAPE(base_y_test, base_pred_test)
#     base_result.append(smape)
#     print('smape: ', smape)

    acc = accuracy(base_y_test, base_pred_test)
#     base_result.append(acc)
    print('acc: ', acc)
    metrics = ['mse', 'rmse', 'mae', 'smape', 'acc']
#     return model, metrics, val_result,base_val_result, test_result, base_result, y_test, base_pred_test, pred_test
    df_for_training['pred'] = np.nan
    df_for_training['pred'].loc[test_idx] = pred_test
    return  df_for_training.loc[test_idx][['sku', target_window, 'pred', base_feature]], model, Xbar.columns.values


def get_results(datasets, target_window, week_num, seasonal_idx):
    '''
    输入：三个sku类别分别的dataset，target_window指rolling周平均还是每日，weeknum是提前的周数, seasonal_idx是dataset lst 里面seasonal sku的idx，不带入product feature
    输出：
    - list of list of results[longtail result, normal result, seasonal result]
    - model：训练出来的model
    - feature_lst： 用于画重要特征图
    
    '''

    test_df_results = []
    model = []
    feature_names_lst = []
    for available_time in tqdm(range(1, week_num+1)):
#     for available_time in tqdm(range(1, 2)):
        label = []
        base_pred = []
        pred = []
        test_dfs = []
        for i in range(len(datasets)):
            dataset = datasets[i]
#             print('dataset sku num before model:', dataset.sku.nunique())
            if i == seasonal_idx:
                test_df, m, feature_names_ = MODEL(available_time, target_window, dataset = dataset, product = False)
            else:
                test_df, m, feature_names_ = MODEL(available_time, target_window, dataset = dataset, product = True)
            model.append(m)
            test_df = test_df.groupby('sku').sum().reset_index()
            test_dfs.append(pd.DataFrame(test_df))
            feature_names_lst.append(feature_names_)
        test_df_result = pd.concat(test_dfs, axis = 0)   
#         end_offer_date_check = product[pd.to_datetime(product.END_OFFER_DATE)>=pd.to_datetime('2019-11-05')].PRODUCT_CODE
#         test_df_result = test_df_result[test_df_result.sku.isin(end_offer_date_check)]
#         print('pred_file qty num: ', test_df_result.qty.sum())
        test_df_result['week_num'] = available_time
        
        
        # for city only
        test_df_result = test_df_result.rename(columns = {str(available_time) + 'w_ago_4w_sum_qty': 'benchmark'})

        
        
        test_df_results.append(test_df_result)
        model.append(m)



    return test_df_results,model,feature_names_lst


def accuracy(y_true, y_pred):
    '''
    input:
    y_true: an array of values of predicted y in different city
    
    output:
    accuracy for that week
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    a = sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))])
    b = sum([max(y_true[i], y_pred[i]) for i in range(len(y_true))])
#     return a, b
    return 1 - a/b

def get_whole_vertical_report(preds, tm_preds):
    '''
    转换预测结果为一个file [sku，qty，benchmark， pred]
    此处qty为label， pred为预测量
    '''

    reports = []
    for i in range(13):
        BJ = preds[0][i].reset_index(drop = True).copy()
        BJ['DIGITAL_HUB'] = 'BJ'
        YD = preds[1][i].reset_index(drop = True).copy()
        YD['DIGITAL_HUB'] = 'YD'
        GF = preds[2][i].reset_index(drop = True).copy()
        GF['DIGITAL_HUB'] = 'GF'
        report_com = pd.concat([BJ,YD,GF], axis = 0)
        report_com['CHANNEL']='com'
#         print('report_com qty num: ' ,report_com.qty.sum())
        BJ2 = tm_preds[0][i].reset_index(drop = True).copy()
        BJ2['DIGITAL_HUB'] = 'BJ'
        YD2 = tm_preds[1][i].reset_index(drop = True).copy()
        YD2['DIGITAL_HUB'] = 'YD'
        GF2 = tm_preds[2][i].reset_index(drop = True).copy()
        GF2['DIGITAL_HUB'] = 'GF'
        report_tm = pd.concat([BJ2,YD2,GF2], axis = 0)
        report_tm['CHANNEL'] = 'tmall'
#         print('report_tm qty num: ' ,report_tm.qty.sum())
        report = pd.concat([report_tm, report_com], axis = 0).reset_index(drop = True)
#         print('report qty num: ' ,report.qty.sum())
        report['pred'] = report['pred'].apply(lambda x: 0 if x<0 else x)
        reports.append(report.fillna(0))
    return reports



def main():
    platform == 'com'
    file_names = ['./features/BJ_com_feature.csv',
                './features/YD_com_feature.csv',
                './features/GF_com_feature.csv']
    target_window = 'qty'


    # preds = get_preds(file_names, platform = 'tmall')

    true = []
    base = []
    pred = []
    preds = []
    names = []
    models = []
    for place in tqdm(file_names):
        name = place[:2]
        names.append(name)
        df = pd.read_csv(place)
        df = feature_engineering(df)

        df_longtail_holiday, df_normal_holiday, df_seasonal_holiday = filter_season_join(df, channel = platform)
        datasets = [df_longtail_holiday, df_normal_holiday, df_seasonal_holiday]
    #     labels, base_preds, preds = get_results(datasets, target_window)
    #     true.append(labels)
    #     base.append(base_preds)
    #     pred.append(preds)
        test_df_results, model, feature_names_lst = get_results(datasets, target_window, week_num = 13)
        models.append(model)
        preds.append(test_df_results)


    platform == 'tmall'
    tm_file_names = ['./features/BJ_tmall_feature.csv',
                    './features/YD_tmall_feature.csv',
                    './features/GF_tmall_feature.csv'
    ]

    target_window = 'qty'

    tm_preds = []
    tm_models = []
    tm_names = []
    for place in tqdm(tm_file_names):
        name = place[:2]

        tm_names.append(name)

        df = pd.read_csv(place)
        df = feature_engineering(df)

        df_longtail_holiday, df_normal_holiday, df_seasonal_holiday = filter_season_join(df, channel = platform)
        datasets = [df_longtail_holiday, df_normal_holiday, df_seasonal_holiday]
        test_df_results, model, feature_names_lst = get_results(datasets, target_window, week_num = 13)
        tm_models.append(model)
        tm_preds.append(test_df_results)

    
    print(platform, ' completed!')
    reports = get_whole_vertical_report(preds, tm_preds)
    for i in range(len(reports)):
        # 改名：更新后qty成为预测量，label更名为label
        reports[i].rename(columns = {'qty': 'label', 'pred':'qty'}, inplace = True)
        reports[i].to_csv('./prediction/hub_prediction_11.5_11.18_sum/' + str(i+1) +'w_ahead_v4.csv')
    # print(report[0].sku.nunique())
    print('done')


main()