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

n_jobs = 5
available_time = 13
predict = True

com_city_dir_path = '/backup/ND/label_files/city_level/com_file_zhipeng/'
tmall_city_dir_path = '/backup/ND/label_files/city_level/tmall_file_zhipeng/'
feature_dir_path = '/backup/ND/feature_files/city_level/feature_eng_yzp/'



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
------------------------------------------Main Model---------------------------------------------
'''
def MODEL( target_window, dataset, product, predict = False):

    available_feature = str(available_time) + 'w_ago_1w_sum_qty'
    base_feature = str(available_time) + 'w_ago_4w_sum_qty'
    num_features = [
        # traffic features  
        'daily_traffic_1wmean', 'daily_traffic_1wstd',  'daily_buyer_count_1wstd', 'daily_buyer_count_1wmean',
        'daily_buyer_count', 'daily_traffic', 
        
        # pdp features
        'MAX_DISCOUNT_PCT', 'MEAN_DISCOUNT_PCT', 'MEAN_CFP_INCL_TAX_LC',
        # product features
        'sku_type_count',
        
        # time labels
        'dayofweek', 'dayofyear', 'year',
        # decayed due to time complexity
#         'holiday_after', 'holiday_before',
        #trend labels
        '365d_ago_1w_mean',  base_feature, '90d_ago_1w_mean', '90d_ago_1w_std'
                ]
    for i in range(available_time, 14):
        num_features.append(str(i) + 'w_ago_1w_sum_qty')
        
    cat_features = [ 
        #product
        # , 
#         'cluster_price','cluster_customer','cluster_popularity', 'sales_volume',
        'KIDS_AGE', 'LAUNCH_TIER','GENDER_CODE','FIT_CODE', 
        
        #time
        'day_desc'

    ]
    if product == True:
        for feature in ['CATEGORY_CODE']:
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
#         print(df3[num_features].isnull().any().any())
#         print(df3[cat_features].isnull().any().any())
        return df3

    df_for_training = fill_dataset(dataset)
    df_for_training = df_for_training.reset_index()


    X = df_for_training.iloc[:,4:].copy()
    Y = df_for_training[[target_window]].copy()
    Y.fillna(0, inplace = True)

    cat_dfs = []
    for feature in cat_features:
        dummies = pd.get_dummies(X[feature], prefix=feature)
        cat_dfs.append(dummies)

    X1 = pd.concat(cat_dfs, axis = 1)
#     print('categorical features encoding done!')

    from sklearn.preprocessing import MinMaxScaler

    mms = MinMaxScaler()

        
    X[num_features] = mms.fit_transform(X[num_features])
    X2 = X.loc[:, num_features]

#     print('mms done.')

    Xbar = pd.concat([X1, X2], axis = 1)

    from sklearn.model_selection import train_test_split
    

    test_set_date = ['2019-11-05','2019-11-06','2019-11-07',
            '2019-11-08','2019-11-09','2019-11-10','2019-11-11','2019-11-12','2019-11-13','2019-11-14','2019-11-15',
               '2019-11-16', '2019-11-17', '2019-11-18' ]
    
    pred_sku_list = get_sku_list()
    pred_date = ['2020-11-05','2020-11-06','2020-11-07',
            '2020-11-08','2020-11-09','2020-11-10','2020-11-11','2020-11-12','2020-11-13','2020-11-14','2020-11-15',
               '2020-11-16', '2020-11-17', '2020-11-18' ]

    df_for_training.date = df_for_training.date.astype(str)
    if predict == False:
        
        test_idx = df_for_training.reset_index()[df_for_training.reset_index().date.isin(test_date)].index
        x_test = Xbar.loc[test_idx]
        y_test = Y.loc[test_idx]
        x = Xbar.drop(test_idx, axis = 0).copy()
        y = Y.drop(test_idx, axis = 0).copy()
    
    else:
        
        final_pred_idx = df_for_training.reset_index()\
                        [(df_for_training.reset_index().date.isin(pred_date))&(df_for_training.sku.isin(pred_sku_list))]\
                        .index
        x_pred = Xbar.loc[final_pred_idx]
        y_pred = Y.loc[final_pred_idx]
#     print('y_test:', y_test.sum())

        x = Xbar.drop(final_pred_idx, axis = 0).copy()
        y = Y.drop(final_pred_idx, axis = 0).copy()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder 


    # train val sets
    val_random_state = 42

    x_train, x_val, y_train, y_val = train_test_split(x,y.values, random_state = val_random_state, test_size = 0.1)

    import xgboost as xgb
    params = {
        'learning_rate':0.015,
        'num_leaves':2**10,
        'max_depth' : 7
        
    }
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
#     pred = model.predict(x_val)


    from sklearn.metrics import mean_squared_error, mean_absolute_error

#     val_result = []
 
    # base val
#     base_y_val = y_val
#     base_pred_val = mms.inverse_transform(x_val[num_features]).T[num_features.index(base_feature)]
    
#     base_val_result = []
    if predict == False:
        pred_test = model.predict(x_test)
    #     test_result = []

    #     print('test scores')
    #     mse = mean_squared_error(y_test, pred_test)
    #     test_result.append(mse)
    #     print('mse: ', mse)

    #     acc = accuracy(y_test, pred_test)
    #     test_result.append(acc)
    #     print('acc: ', acc)

    #     base_y_test = y_test
    #     base_pred_test = mms.inverse_transform(x_test[num_features]).T[num_features.index(base_feature)]
    #     base_result = []

    #     mse = mean_squared_error(base_y_test, base_pred_test)
    #     print('test base')
    #     print('mse: ', mse)

    #     acc = accuracy(base_y_test, base_pred_test)

    #     print('acc: ', acc)
    #     metrics = ['mse', 'rmse', 'mae', 'smape', 'acc']
        df_for_training['pred'] = np.nan
        df_for_training['pred'].loc[test_idx] = pred_test

        return  df_for_training.loc[test_idx][['sku', target_window, 'pred', base_feature]], model, Xbar.columns.values
    elif predict == True:
        pred_test = model.predict(x_pred)
        df_for_training['pred'] = np.nan
        df_for_training['pred'].loc[final_pred_idx] = pred_test
        return  df_for_training.loc[final_pred_idx][['sku', target_window, 'pred', base_feature]], model, Xbar.columns.values

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



def get_one_result(city, available_time=available_time, predict = predict):
        df_longtail_holiday = pd.read_csv(feature_dir_path + 'longtail_' + city ) 
        df_normal_holiday = pd.read_csv(feature_dir_path + 'normal_' + city )
        df_seasonal_holiday = pd.read_csv(feature_dir_path + 'seasonal_' + city )

        label = []
        base_pred = []
        pred = []
        test_dfs = []
        feature_names_lst = []
        model = []
        datasets = [df_normal_holiday, df_longtail_holiday, df_seasonal_holiday]
        
        for i in range(len(datasets)):
            dataset = datasets[i]
            if dataset.qty.isnull().mean() < 0.95:
                
        #             print('dataset sku num before model:', dataset.sku.nunique())
                if i == 0:
                    test_df, m, feature_names_ = MODEL( target_window = 'qty', dataset = dataset, product = True, predict = predict)
                else:
                    test_df, m, feature_names_ = MODEL( target_window = 'qty', dataset = dataset, product = False, predict = predict)
                test_df.pred = test_df.pred.apply(lambda x: max(0, round(x)))
                test_df = test_df.groupby('sku').sum().reset_index()
                test_dfs.append(pd.DataFrame(test_df))
                model.append(m)
                feature_names_lst.append(feature_names_)
        if len(test_dfs) != 0:
            test_df_result = pd.concat(test_dfs, axis = 0)   
        
            assert len(test_df_result) != 0

            test_df_result['week_num'] = available_time

            # for city only
            test_df_result = test_df_result.rename(columns = {str(available_time) + 'w_ago_4w_sum_qty': 'benchmark'})
            ''' 
            model储存路径在以下更改
            '''
            test_df_result.to_csv(f'/ND_result_yzp/{city}')
        else:
            print(f'too small to predict: {city} : {datasets[0].qty.sum()}')



'''
---------------------------------------Run----------------------------------------
'''
def main():
    to_pred = pd.read_csv('/home/yh/11/info/city_list.csv', header = None)
    mappath = '/backup/POC_Data/cityMapping.csv'
    citymap = pd.read_csv(mappath)
    city2province = citymap.Province.to_dict()
    pinyin2id = {v:k for k,v in citymap.City.to_dict().items()}
    pred_city_id = [pinyin2id[city] for city in to_pred[1].values]

    coms = [name for name in os.listdir('/backup/ND/label_files/city_level/com_file_zhipeng')]
    # coms = [str(city) + '_com.csv' for city in pred_city_id]

    for city in tqdm(coms):
        # print(city)
        get_one_result(city, available_time = available_time)
    
    tmalls = [name for name in os.listdir('/backup/ND/label_files/city_level/tmall_file_zhipeng')]

    # tmalls = [str(city) + '_tmall.csv' for city in pred_city_id]

    for city in tqdm(tmalls):
        get_one_result(city,  available_time = available_time)


def main_province_parallel():

    coms = [name for name in os.listdir('/backup/ND/label_files/city_level/com_file_zhipeng')]
    # coms = [str(city) + '_com.csv' for city in pred_city_id]
    print('working on com')
    func = get_one_result
    funclist = [get_one_result] * len(coms)
    Parallel(n_jobs=n_jobs)(delayed(get_one_result)(name) for  name in coms)
    tmalls = [name for name in os.listdir('/backup/ND/label_files/city_level/tmall_file_zhipeng')]
    # tmalls = [str(city) + '_tmall.csv' for city in pred_city_id]
    print('tmall started')
    # Parallel(n_jobs=15)(delayed(Parallel)(func, name) for func, name in zip(funclist, tmalls))
    Parallel(n_jobs=n_jobs)(delayed(get_one_result)(city) for city in tmalls)


main_province_parallel()
