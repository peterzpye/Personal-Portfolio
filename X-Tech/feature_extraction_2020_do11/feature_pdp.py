import pandas as pd
import numpy as np
import os
import tqdm

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm


def main():
    path = '/backup/ND/3.1.3.21.2 Digital Tmall PDP/'
    df = pd.read_csv(path + 'pdp_sum.csv')
    df.groupby('tran_dt').sum()[['pv','pdp_traffic']]\
    .reset_index().rename(columns = {'tran_dt':'date'})\
    .to_csv('/backup/ND/label_files/city_level/pdp_sum_zhipeng.csv', index = False)
    print('done!')
