import numpy as np
import pandas as pd
import seaborn as sb
import os
from tqdm import tqdm

def get_je_sku_list():
    '''
    获取je 的6月最后一周的sku
    :return:
    '''
    import hashlib
    file_path = '/backup/POC_Data/20191010--20191120.xlsx'
    je_df = pd.read_excel(file_path)
    #print(je_df.create_datetime[0])
    je_df['create_datetime'] = je_df.create_datetime.apply(lambda x: str(x).split()[0])
    spec_dates = [f'2019-11-{i}' for i in range(5,18)]
    res_df = je_df[je_df.create_datetime.isin(spec_dates)]
    res_df = res_df.groupby('product_code').agg({'quantity':'sum'}).reset_index()
    res_df['product_code'] = res_df.product_code.apply(lambda c: hashlib.sha256(c.encode()).hexdigest())
    #print(res_df)
    sku_list_len = res_df.product_code.nunique()
    print(sku_list_len)
    res_df = res_df.rename(columns = {'product_code':'sku', 'quantity': 'je_benchmark'})
    res_df['je_benchmark'] = res_df['je_benchmark']/2
    return res_df


def get_pleftovers(df):
    return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: abs(row['qty'] - row['label']), axis = 1).sum()
#     return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: max((row['qty'] - row['label']),0) , axis = 1).sum()

def get_bleftovers(df, je = je):
#     je = get_je_sku_list()
    df = df.merge(je, on = 'sku')
    return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: abs(row['je_benchmark'] - row['label']), axis = 1).sum()
#     return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: max((row['je_benchmark'] - row['label']),0) , axis = 1).sum()

def get_ppleftovers(df, hub):
    df = df[df.DIGITAL_HUB == hub].copy()
    return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: abs(row['qty'] - row['label']), axis = 1).sum()
#     return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: max((row['qty'] - row['label']),0) , axis = 1).sum()
def get_pbleftovers(df, hub):
#     je = get_je_sku_list()
    df = df.merge(je, on = 'sku')
    df = df[df.DIGITAL_HUB == hub].copy()
    return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: abs(row['je_benchmark'] - row['label']), axis = 1).sum()
#     return df.groupby(['sku', 'CHANNEL']).sum().apply(lambda row: max((row['je_benchmark'] - row['label']),0) , axis = 1).sum()




def main():
    
    je = get_je_sku_list()
    totals = []
    tmalls = []
    coms = []
    dfs = []
    tm_BJs = []
    tm_YDs = []
    tm_GFs = []
    com_BJs = []
    com_YDs = []
    com_GFs = []
    com_bases = []
    tm_bases = []
    week_num = []
    for i in range(1, 14):
        week_num.append(i)
        df = pd.read_csv('./prediction/hub_prediction_11.5_11.18_sum/' + str(i) +'w_ahead_v3.csv')
        df_leftover = df[df.sku.isin(je.sku)]
        dfs.append(df_leftover)
        tm_df = df_leftover[df_leftover.CHANNEL == 'tmall'].copy()
        com_df = df_leftover[df_leftover.CHANNEL == 'com'].copy()
        tm = get_pleftovers(tm_df)
        tm_base = get_bleftovers(tm_df)
        tm_bases.append(tm_base)
        tm_BJs.append(get_ppleftovers(tm_df, 'BJ'))
        tm_YDs.append(get_ppleftovers(tm_df, 'YD'))
        tm_GFs.append(get_ppleftovers(tm_df, 'GF'))
        
        com = get_pleftovers(com_df)
        com_BJs.append(get_ppleftovers(com_df, 'BJ'))
        com_YDs.append(get_ppleftovers(com_df, 'YD'))
        com_GFs.append(get_ppleftovers(com_df, 'GF'))
        com_base = get_bleftovers(com_df)
        com_bases.append(com_base)
        tmalls.append(tm)
        coms.append(com)
        totals.append(tm+com)
    leftovers = pd.DataFrame({'leftover_qty_total': totals, 'leftover_qty_tmall':tmalls,
                        'leftover_tmall_BJ': tm_BJs, 'leftover_tmall_YD': tm_YDs,'leftover_tmall_GF': tm_GFs,
                        'leftover_qty_com':coms, 'leftover_com_BJ': com_BJs, 'leftover_com_YD': com_YDs,
                        'leftover_com_GF': com_GFs, 'base_com':com_bases, 'base_tmall': tm_bases,
                        'week_num':week_num})
    return leftovers