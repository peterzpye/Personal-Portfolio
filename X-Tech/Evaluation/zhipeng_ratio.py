import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from tqdm import tqdm



def get_pratio(df, channel):
    df = df[df.CHANNEL == channel].copy()
    BJ = df[df.DIGITAL_HUB == 'BJ'][['sku', 'qty']].set_index('sku')
    BJ.columns = ['BJ']
    YD = df[df.DIGITAL_HUB == 'YD'][['sku', 'qty']].set_index('sku')
    YD.columns = ['YD']
    GF = df[df.DIGITAL_HUB == 'GF'][['sku', 'qty']].set_index('sku')
    GF.columns = ['GF']
    t = pd.concat([BJ,YD,GF], axis = 1)
    t = t[['BJ','YD', 'GF']]
    return '/'.join(str(np.round(x*100, 1)) for x in (t.sum().values/t.sum().sum()).tolist())
#     return t.sum()/t.sum().sum()
#     return t

def get_bratio(df, channel):
    df = df[df.CHANNEL == channel].copy()
    BJ = df[df.DIGITAL_HUB == 'BJ'][['sku', 'benchmark']].set_index('sku')
    BJ.columns = ['BJ']
    YD = df[df.DIGITAL_HUB == 'YD'][['sku', 'benchmark']].set_index('sku')
    YD.columns = ['YD']
    GF = df[df.DIGITAL_HUB == 'GF'][['sku', 'benchmark']].set_index('sku')
    GF.columns = ['GF']
    t = pd.concat([BJ,YD,GF], axis = 1)
    t = t[['BJ', 'YD', 'GF']]
    return '/'.join(str(np.round(x*100, 1)) for x in (t.sum().values/t.sum().sum()).tolist())
#     return t.sum()/t.sum().sum()
#     return t

def get_lratio(df, channel):
    df = df[df.CHANNEL == channel].copy()
    BJ = df[df.DIGITAL_HUB == 'BJ'][['sku', 'label']].set_index('sku')
    BJ.columns = ['BJ']
    YD = df[df.DIGITAL_HUB == 'YD'][['sku', 'label']].set_index('sku')
    YD.columns = ['YD']
    GF = df[df.DIGITAL_HUB == 'GF'][['sku', 'label']].set_index('sku')
    GF.columns = ['GF']
    t = pd.concat([BJ,YD,GF], axis = 1)
    t.fillna(0,inplace = True)
    t = t[['BJ', 'YD', 'GF']]
    return '/'.join(str(np.round(x*100, 1)) for x in (t.sum().values/t.sum().sum()).tolist())


def main():
    tsplit_ratio = []
    tbase_ratio = []
    tlabel_split_ratio = []
    csplit_ratio = []
    cbase_ratio = []
    clabel_split_ratio = []

    for i in range(1,14):
        #使用不同的版本
        df = pd.read_csv('./prediction/hub_prediction_11.5_11.18_sum/' + str(i) +'w_ahead_v4.csv')

        tsplit_ratio.append(get_pratio(df, channel = 'tmall'))
        tbase_ratio.append(get_bratio(df,  channel = 'tmall'))
        tlabel_split_ratio.append(get_lratio(df,  channel = 'tmall'))
        csplit_ratio.append(get_pratio(df, channel = 'com'))
        cbase_ratio.append(get_bratio(df,  channel = 'com'))
        clabel_split_ratio.append(get_lratio(df,  channel = 'com'))
    ratio_report = pd.DataFrame({'tmall_split_ratio':tsplit_ratio, 'tmall_base_ratio': tbase_ratio, 'tmall_label_ratio': tlabel_split_ratio,
                'com_split_ratio':csplit_ratio, 'com_base_ratio': cbase_ratio, 'com_label_ratio': clabel_split_ratio}).T
    return ratio_report
