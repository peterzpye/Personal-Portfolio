import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_traffic_features(df):
    daily_df = df.groupby('date').sum()[[ 'traffic', 'buyer_count', 'demand']]
    daily_df.columns = ['daily_traffic', 'daily_buyer_count', 'daily_demand']
    
    monthly_df = df.groupby('tran_dt_by_month').sum()[['traffic', 'buyer_count', 'demand']].reset_index()
    monthly_df.columns = ['tran_dt_by_month','monthly_traffic', 'monthly_buyer_count', 'monthly_demand']

    daily_traffic_channel = df.groupby('date').apply(lambda x: x.sort_values(by ='traffic').tail(1)).reset_index(level = 1, drop = True)
    
    daily_traffic_channel = daily_traffic_channel[[ 'device', 'channel']]

    final_df = pd.concat([daily_df, daily_traffic_channel], axis = 1)
    final_df['date'] = final_df.index
    final_df = final_df.reset_index(drop = True)
    final_df['tran_dt_by_month'] = final_df.date.apply(lambda x: x[:-3])
    final_df = final_df.merge(monthly_df, on = 'tran_dt_by_month', how = 'left')

    return final_df

def main():
    get_traffic_features(df).to_csv('/backup/ND/feature_files/traffic_feature_yzp.csv', index = False)