'''
Author: Zhipeng Ye



Error Distribution:
0-1: 102343,
1-1: 32500,
2-3: 6258,
3-4: 943

Improvement from last hw3: 0.994 -->
RMSE:0.9825312515652006 (Val score)

Excecution Time: 1253s

Description:
This model is a hybrid model applying mainly feature combination and feature augmentations. It combines the features of user and restaurants(Content based features) from business.json file and user.json file, and also collected collaborative filtering methods such as average pearson correlation for each users and average item scores. I attempted using SVD, pLSA, and LDA from the Collaborative Filtering strategy as additional features but the model performance was greatly decreased and was therefore excluded from the code below. Besides, the model also derived features from photos, checkin, and tips.


Besides, it also used meta-level strategy as well. I trained a single model with only content features, and passed it into the main model as a feature, which was able to help me improve a bit. For intermediate model, I used XGBoost Regressor, with more complex hyperparameters. The main model I used is also XGBoost Regressor, and applied very strict regularization. I used 200 in L2 regularization, trained only 70 trees, used 2 as my gamma, and used 0.15 as my learning rate. I also applied feature scaling to assist in training stage.


I did not use switching method and collaborative filtering model for mainly two reasons: they are inferior in performance and they have huge in-sample out-of-sample difference. While I have item-based CF and user-based CF, they exposed huge performance difference in training set and testing set and would cause the model to greatly overfit, and was therefore not included in this method. I also tried switching method, but combining with CF method would only decrease my model performance in any threshold. They are therefore not included in my final model.

'''


from pyspark import SparkContext, SparkConf
conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")

sc = SparkContext.getOrCreate(conf = conf)
# sc = SparkContext.getOrCreate()

sc.setLogLevel("WARN")

import gc
import numpy as np
from sklearn.impute import SimpleImputer
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import csv
def write_csv(res, path):
    with open(path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['user_id', 'business_id', 'prediction'])
        for pair in res:
            csv_writer.writerow(pair)
    print('csv writing finished')

class CategoricalEncoder:
    def __init__(self):
        from collections import Counter
        self.feature_names = []

    def fit(self, data):
        self.feature_names = sorted(data)

    def transform(self, data):
        res = [0] * len(self.feature_names)

        for j in range(len(self.feature_names)):
            if self.feature_names[j] in set(data):
                res[j] = 1
        return res


def pearson_similarity(a, b):
    '''
    pearson correlation between a and b
    '''
    if len(a) != len(b):
        raise ValueError(f'{a}, {b} length not equal to each other')
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    numerator = np.sum((a - a_mean) * (b - b_mean))
    denomenator = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    if denomenator > 0:
        coeff = numerator / denomenator
        return coeff
    return 0


def derive_sim_feat(x):
    uid, bid = x[0], x[1]
    tho = 2.5
    score = x[2]
    items = user_rating_info[uid]
    cands = []
    for item in items:
        if item != bid:
            common_users = list(rater_info[item].intersection(rater_info[bid]))
            a = [score_info[(user, item)] for user in common_users]
            b = [score_info[(user, bid)] for user in common_users]
            try:
                sim = pearson_similarity(a, b)
            except:
                sim = 0
            sim = sim * abs(sim) ** (tho)
            cands.append(sim)
    if len(cands) > 0:
        feats = (np.mean(cands), np.std(cands), np.min(cands), np.max(cands))
    else:
        feats = (0, 0, 0, 0)
    return (uid, feats)


def derive_photo_features(photo_path):
    #     sia = SentimentIntensityAnalyzer()
    photo_rdd = sc.textFile(photo_path).map(lambda x: json.loads(x)).persist()
    all_photo_types = photo_rdd.map(lambda x: x['label']).distinct().collect()

    def derive_photo_features_(rdd):
        label_cnt = {k: 0 for k in all_photo_types}
        p_s, n_s, neu_s, c_s = [], [], [], []
        cnt = 0
        for instance in rdd:
            label = instance[0]
            label_cnt[label] += 1
            cnt += 1
        return tuple([cnt]) + tuple(
            [item[1] for item in sorted(label_cnt.items(), key=lambda x: x[0])] + [max(label_cnt.values())])

    photo_feats_info = photo_rdd.map(lambda x: (x['business_id'], (x['label'], x['caption']))).groupByKey() \
        .mapValues(derive_photo_features_).collectAsMap()
    return photo_feats_info


def derive_business_features(business_fpath):
    def get_city_avg_info_(business_fpath):
        business_rdd = sc.textFile(business_fpath).map(lambda x: json.loads(x))
        city_avg_info = business_rdd.map(lambda x: (x['city'], x['stars'])).groupByKey() \
            .mapValues(list).mapValues(np.mean).collectAsMap()
        return city_avg_info

    def get_state_avg_info_(business_fpath):
        business_rdd = sc.textFile(business_fpath).map(lambda x: json.loads(x))
        state_avg_info = business_rdd.map(lambda x: (x['state'], x['stars'])).groupByKey() \
            .mapValues(list).mapValues(np.mean).collectAsMap()
        return state_avg_info

    def derive_tip_b_features(tip_path):
        all_item_set = set(train_rdd.map(lambda x: x[1]).collect()).union(set(test_rdd.map(lambda x: x[1]).collect()))
        tip_user_feats_info = sc.textFile(tip_path).map(lambda x: json.loads(x)).map(
            lambda x: (x['business_id'], x['likes'])).groupByKey().mapValues(list).map(
            lambda x: (x[0], (len(x[1]), sum(x[1])))).filter(lambda x: x[0] in all_item_set).collectAsMap()
        return tip_user_feats_info

    def derive_business_features_(row):
        if row['attributes'] == None:
            row['attributes'] = {}
        bid = row['business_id']
        city_avg = city_avg_info[row['city']]
        state_avg = state_avg_info[row['state']]

        alcohol = int(bool(row['attributes'].get('Alcohol', -1)))
        dogs = int(bool(row['attributes'].get('DogsAllowed', -1)))
        cater = int(bool(row['attributes'].get('Caters', -1)))
        goodforkid = int(bool(row['attributes'].get('GoodForKids', -1)))
        outdoor = int(bool(row['attributes'].get('OutdoorSeating', -1)))
        review_count = int(row.get('review_count'))
        star = int(row.get('stars', -1))
        is_open = row.get('is_open', -1)
        price_range = int(row['attributes'].get('RestaurantsPriceRange2', -1))
        goodforgroup = int(bool(row['attributes'].get('RestaurantsGoodForGroups', -1)))
        delivery = int(bool(row['attributes'].get('RestaurantsDelivery', -1)))
        reserve = int(bool(row['attributes'].get('RestaurantsReservations', -1)))
        #         ambience = row['attributes'].get('Ambience', '{}')
        #         amb_cnt = len(ambience.values())
        takeout = int(bool(row['attributes'].get('RestaurantsTakeOut', -1)))
        table_service = int(bool(row['attributes'].get('RestaurantsTableService', -1)))
        tv = int(bool(row['attributes'].get('HasTV', -1)))
        open_info = row.get('hours', {})
        if open_info is None:
            open_info = {}
            open_days = np.nan
            open_hours = np.nan
        else:
            open_info = list(open_info.values())
            open_days = len(open_info)
            open_info = [x.split('-') for x in open_info]
            open_hours = sum([int(val[1].split(':')[0]) - int(val[0].split(':')[0]) for val in open_info])
        long = row.get('longitude', np.nan)
        lat = row.get('latitude', np.nan)
        try:
            category_lst = row.get('categories', '').split(',')
        except:
            category_lst = []

        cred_cards = int(bool(row['attributes'].get('BusinessAcceptsCreditCards', 0)))
        attire = [row['attributes'].get('RestaurantsAttire', 'NA')]
        attire_encoded = attire_encoder.transform(attire)
        noise = [row['attributes'].get('NoiseLevel', 'NA')]
        noise_encoded = noise_encoder.transform(noise)
        category_encoded = category_encoder.transform(list(category_lst))
        category_length = len(category_lst)

        tip_feats = tip_b_feats_info.get(bid, tuple([np.nan] * len(list(tip_b_feats_info.values())[0])))
        feats = [star, \
                 review_count, \
                 #                              star*review_count, \
                 #                              np.log(int(review_count)), \
                 city_avg, \
                 state_avg, \
                 is_open, \
                 delivery, \
                 #                              amb_cnt, \
                 reserve, \
                 tv, \
                 sum([price_range, goodforgroup, alcohol, outdoor, goodforkid, dogs, cater, cred_cards]), \
                 price_range, \
                 goodforgroup, \
                 alcohol, \
                 outdoor, \
                 goodforkid, \
                 dogs, \
                 cater, \
                 cred_cards, \
                 category_length, \
                 open_hours, \
                 open_days, \
                 open_hours / open_days, \
                 #                              long, \
                 #                              lat
                 ]
        feats = [x if x != -1 else np.nan for x in feats]
        return (row['business_id'], \
                tuple(feats +
                      attire_encoded +
                      noise_encoded +
                      category_encoded +
                      list(tip_feats)))

    business_cats = sc.textFile(business_fpath, 12).map(lambda x: json.loads(x)).map(
        lambda x: x.get('categories', None)) \
        .filter(lambda x: x != None) \
        .flatMap(lambda x: x.split(',')) \
        .map(lambda x: (x.strip(), 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda x: -x[1]) \
        .map(lambda x: x[0]) \
        .take(20)
    category_encoder = CategoricalEncoder()
    category_encoder.fit(business_cats)
    del business_cats

    business_attires = sc.textFile(business_fpath, 12).map(lambda x: json.loads(x)).map(
        lambda x: (x['attributes'] if x['attributes'] != None else {}).get('RestaurantsAttire', 'NA')) \
        .distinct().collect()
    attire_encoder = CategoricalEncoder()
    attire_encoder.fit(business_attires)
    del business_attires

    business_noise = sc.textFile(business_fpath, 12).map(lambda x: json.loads(x)).map(
        lambda x: (x['attributes'] if x['attributes'] != None else {}).get('NoiseLevel', 'NA')) \
        .distinct().collect()
    noise_encoder = CategoricalEncoder()
    noise_encoder.fit(business_noise)
    del business_noise
    gc.collect()

    city_avg_info = get_city_avg_info_(business_fpath)
    state_avg_info = get_state_avg_info_(business_fpath)
    tip_b_feats_info = derive_tip_b_features(tip_path)
    #     business_star_mean = np.mean([x['stars'] for x in sc.textFile(business_fpath, 12).map(lambda x: json.loads(x)).collect()])
    business_feats_info = sc.textFile(business_fpath, 12).map(lambda row: json.loads(row)) \
        .map(lambda row: derive_business_features_(row)) \
        .collectAsMap()
    return business_feats_info


def derive_checkin_features(checkin_path):
    def derive_checkin_features_(rdd):
        bid = rdd['business_id']
        count = len(rdd.get('time', {}).values())
        checkin_sum = sum(list(rdd.get('time', {}).values()))
        return (bid, (count, checkin_sum, checkin_sum / count, checkin_sum ** 2))

    checkin_feats_info = sc.textFile(checkin_path).map(lambda row: json.loads(row)).map(
        derive_checkin_features_).collectAsMap()
    return checkin_feats_info


def derive_user_features(user_fpath):
    def derive_tip_user_features(tip_path):
        all_user_set = set(train_rdd.map(lambda x: x[0]).collect()).union(set(test_rdd.map(lambda x: x[0]).collect()))
        tip_user_feats_info = sc.textFile(tip_path).map(lambda x: json.loads(x)).map(
            lambda x: (x['user_id'], x['likes'])).groupByKey().mapValues(list).map(
            lambda x: (x[0], (len(x[1]), sum(x[1])))).filter(lambda x: x[0] in all_user_set).collectAsMap()
        return tip_user_feats_info

    def derive_user_features_(rdd):
        uid = rdd['user_id']
        comm_feat = [x for x in rdd.values() if type(x) != str]
        num_friends = len(rdd.get('friends', '').split(', '))
        yelp_age = 2020 - int(rdd['yelping_since'].split('-')[0])
        tip_feats = tip_user_feats_info.get(uid, tuple([np.nan] * len(list(tip_user_feats_info.values())[0])))
        return (uid, tuple(comm_feat + [num_friends, yelp_age]) + tip_feats)

    tip_user_feats_info = derive_tip_user_features(tip_path)
    user_feature_data_info = sc.textFile(user_fpath, 12).map(lambda x: json.loads(x)).map(
        derive_user_features_).collectAsMap()
    return user_feature_data_info


def calculate_item_info(rdd):
    '''
    calculate item stats: mean and std

    '''
    uid, bid = rdd[0], rdd[1]
    users = rater_info.get(bid, [])
    res = []
    threshold = 0
    for user in users:
        if user != uid:
            res.append(score_info[(user, bid)])
    if len(res) > threshold:
        ans = (np.mean(res), np.std(res), np.median(res), np.sum(res))
    else:
        #         ans = (avg_item_avg, avg_item_std)
        ans = (np.nan, np.nan, np.nan, np.nan)
    return ((uid, bid), ans)


def calculate_user_info(rdd):
    '''
    calculate user stats: mean and std
    '''
    uid, bid = rdd[0], rdd[1]
    threshold = 0
    items = user_rating_info[uid]
    res = []
    for item in items:
        if item != bid:
            res.append(score_info[(uid, item)])

    if len(res) > threshold:
        ans = (np.mean(res), np.std(res), np.median(res), np.sum(res))
    else:
        #         ans = (avg_user_avg, avg_user_std)
        ans = (np.nan, np.nan, np.nan, np.nan)
    return ((uid, bid), ans)


def make_train_set(train_rdd, regression=True):
    def combine(rdd):
        '''
        make the training set
        '''
        uid, bid, score = rdd[0], rdd[1], rdd[2]
        num_raters = rater_count_info.get(bid, np.nan)
        num_rating = user_count_info.get(uid, np.nan)
        #         user_avg, user_median = user_avg_test_info.get(uid, (np.nan, np.nan))
        #         item_avg, item_median = item_avg_test_info.get(bid, (np.nan, np.nan))
        u_cf_feats = user_avg_info.get((uid, bid))
        i_cf_feats = item_avg_info.get((uid, bid))

        business_feat = business_feats_info.get(bid, default_b_val)
        checkin_feat = checkin_feats_info.get(bid, default_c_val)
        u_feats = tuple(user_feature_data_info.get(uid, default_user_val))
        p_feats = photo_feats_info.get(bid, default_p_val)
        sim_feats = user_sim_info.get(uid, default_sim_val)
        CB_score = CB_info.get((uid, bid), np.nan)
        if regression == False:
            if score == 5:
                score = 1
            else:
                score = 0
        base_feat = (score, CB_score, num_raters, num_rating) + u_cf_feats + i_cf_feats
        #         base_feat =( score, num_raters, num_rating, user_avg, item_avg)
        features = base_feat + checkin_feat + u_feats + business_feat + p_feats + sim_feats
        na_pct = len([x for x in features if x == np.nan]) / len(features)
        b_na_pct = len([x for x in business_feat if x == np.nan]) / len(business_feat)
        u_na_pct = len([x for x in u_feats if x == np.nan]) / len(u_feats)
        features += (na_pct, b_na_pct, u_na_pct)
        return ((uid, bid), features)

    res = []
    for rdd in train_rdd.collect():
        res.append(combine(rdd)[1])
    data = np.array(res)
    x, y = data[:, 1:], data[:, 0]
    return x, y


def make_val_set(val_rdd, regression=True, predict = False):
    def combine_test(rdd, predict = False):
        '''
        make the feature space including filling NAN values
        '''
        uid, bid = rdd[0], rdd[1]
        if predict == False:
            score = rdd[2]
        num_raters = rater_count_info.get(bid, np.nan)
        num_rating = user_count_info.get(uid, np.nan)
        u_cf_feats = user_avg_test_info.get(uid, tuple([np.nan] * 4))
        i_cf_feats = item_avg_test_info.get(bid, tuple([np.nan] * 4))

        business_feat = business_feats_info.get(bid, default_b_val)
        checkin_feat = checkin_feats_info.get(bid, default_c_val)
        CB_score = CB_test_info.get((uid, bid), np.nan)
        u_feats = tuple(user_feature_data_info.get(uid, default_user_val))
        p_feats = photo_feats_info.get(bid, default_p_val)
        sim_feats = user_sim_info.get(uid, default_sim_val)
        if regression == False:
            if score == 5:
                score = 1
            else:
                score = 0
        if predict == False:
            base_feat = (score, CB_score, num_raters, num_rating) + u_cf_feats + i_cf_feats
        else:
            base_feat = (CB_score, num_raters, num_rating) + u_cf_feats + i_cf_feats
        #         base_feat =( score, num_raters, num_rating, user_avg, item_avg)
        features = base_feat + checkin_feat + u_feats + business_feat + p_feats + sim_feats
        na_pct = len([x for x in features if x == np.nan]) / len(features)
        b_na_pct = len([x for x in business_feat if x == np.nan]) / len(business_feat)
        u_na_pct = len([x for x in u_feats if x == np.nan]) / len(u_feats)
        features += (na_pct, b_na_pct, u_na_pct)
        return ((uid, bid), features)
    res = []
    for rdd in val_rdd.collect():
        res.append(combine_test(rdd, predict = predict)[1])
    data = np.array(res)
    if predict == False:
        x, y = data[:, 1:], data[:, 0]
        return x, y
    else:
        return data

def build_CB_model():
    def CB_features(rdd, pred = False):
        uid, bid = rdd[0], rdd[1]
        if pred == False:
            score = rdd[2]
        u_feats = user_feature_data_info.get(uid, default_user_val)
        b_feats = business_feats_info.get(bid, default_b_val)
        c_feats = checkin_feats_info.get(bid, default_c_val)
        combined = tuple(u_feats) + tuple(b_feats) + tuple(c_feats)
        if pred == False:
            return ((uid, bid), tuple([score]) + combined)
        else:
            return ((uid, bid), combined)
    def make_data(data_rdd, pred = False):
        res = []
        for rdd in data_rdd.collect():
            if pred == False:
                res.append(CB_features(rdd)[1])
            else:
                res.append(CB_features(rdd, pred = True)[1])
        if pred == False:
            data = np.array(res)
            x, y = data[:, 1:], data[:, 0]
            return x, y
        else:
            x = np.array(res)
            return x
    x,y = make_data(train_rdd)
    x_test = make_data(test_rdd, pred = True)
    # x_test, y_test = make_data(test_rdd)
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.15,
        reg_alpha=0.2,
        max_depth=5,
        booster='dart',
        missing=np.nan,
        # reg_lambda = 0.8,
        n_jobs=-1,
        seed = 666
    )
    model.fit(x,y)
    CB_info = dict(zip(train_rdd.map(lambda x: (x[0], x[1])).collect(), model.predict(x)))
    CB_test_info = dict(zip(test_rdd.map(lambda x: (x[0], x[1])).collect(), model.predict(x_test)))
    # print(np.sqrt(mean_squared_error(y_test, model.predict(x_test))))
    return CB_info, CB_test_info


if __name__ == '__main__':
    import sys
    import time


    start = time.time()
    # folder = 'C:/Users/yzp60/Downloads/DSCI553/HW3/'
    folder = sys.argv[1]

    fpath = folder + '/yelp_train.csv'
    # test_fpath = folder + 'yelp_val.csv'
    test_fpath = sys.argv[2]
    output_path = sys.argv[3]
    business_fpath = folder + '/business.json'
    user_fpath = folder + '/user.json'
    checkin_path = folder + '/checkin.json'
    review_path = folder + '/review_train.json'
    photo_path = folder + '/photo.json'
    tip_path = folder + '/tip.json'





    train_rdd = sc.textFile(fpath, 12).filter(lambda x: x != 'user_id,business_id,stars') \
    .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2]) )).persist()
    test_rdd = sc.textFile(test_fpath, 12).filter(lambda x: x != 'user_id,business_id,stars') \
    .map(lambda x: (x.split(',')[0], x.split(',')[1] ))
    pred_rdd = sc.textFile(test_fpath, 12).filter(lambda x: x != 'user_id,business_id,stars') \
    .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2]) ))

    train_rdd = train_rdd.persist()
    rater_info = train_rdd.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda a, b: a.union(b)).collectAsMap()
    rater_count_info = train_rdd.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda a, b: a.union(b)).map(
        lambda x: (x[0], len(x[1]))).collectAsMap()
    user_count_info = train_rdd.map(lambda x: (x[0], set([x[1]]))).reduceByKey(lambda a, b: a.union(b)).map(
        lambda x: (x[0], len(x[1]))).collectAsMap()
    user_rating_info = train_rdd.map(lambda x: (x[0], set([x[1]]))).reduceByKey(lambda a, b: a.union(b)).collectAsMap()
    score_info = train_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    over_all_mean = train_rdd.map(lambda x: x[2]).mean()
    train_rdd = train_rdd.unpersist()

    avg_rater_count = np.mean(list(rater_count_info.values()))
    avg_rating_count = np.mean(list(user_count_info.values()))

    user_avg_info = train_rdd.map(lambda x: calculate_user_info(x)).collectAsMap()
    item_avg_info = train_rdd.map(lambda x: calculate_item_info(x)).collectAsMap()
    user_avg_test_info = train_rdd.map(lambda x: (x[0], x[2])).groupByKey().mapValues(list).map(
        lambda x: (x[0], (np.mean(x[1]), np.std(x[1]), np.median(x[1]), np.sum(x[1])))).collectAsMap()
    item_avg_test_info = train_rdd.map(lambda x: (x[1], x[2])).groupByKey().mapValues(list).map(
        lambda x: (x[0], (np.mean(x[1]), np.std(x[1]), np.median(x[1]), np.sum(x[1])))).collectAsMap()

    business_feats_info = derive_business_features(business_fpath)
    user_feature_data_info = derive_user_features(user_fpath)
    checkin_feats_info = derive_checkin_features(checkin_path)
    photo_feats_info = derive_photo_features(photo_path)

    default_user_val = tuple([np.nan] * len(list(user_feature_data_info.values())[0]))
    default_b_val = tuple([np.nan] * len(list(business_feats_info.values())[0]))
    default_c_val = tuple([np.nan] * len(list(checkin_feats_info.values())[0]))
    default_p_val = tuple([np.nan] * len(list(photo_feats_info.values())[0]))


    CB_info, CB_test_info = build_CB_model()

    user_sim_info = train_rdd.map(derive_sim_feat).collectAsMap()
    default_sim_val = tuple([np.nan] * len(list(user_sim_info.values())[0]))
    x, y = make_train_set(train_rdd)
    print('training dataset ready')

    x_test = make_val_set(test_rdd, predict=True)
    print('val dataset ready')

    imp = SimpleImputer()
    x = imp.fit_transform(x)
    x_test = imp.transform(x_test)
    ss = StandardScaler()
    ss.fit(x)
    x = ss.transform(x)
    x_test = ss.transform(x_test)

    model = XGBRegressor(
        n_estimators=70,
        learning_rate=0.15,
        reg_alpha=10,
        max_depth=3,
        missing=np.nan,
        subsample=0.7,
        reg_lambda=100,
        n_jobs=-1,
        gamma=2,
        min_child_weight=1,
        #     nthread = -1,
        seed=555
                        )
    model.fit(np.array(x), np.array(y))
    print('model fitted!')
    print(model.get_params())


    # x_test, y_test = make_val_set(test_rdd)
    # y_pred = model.predict(np.array(x_test))
    # rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    # print('oob rmse is : ', rmse)


    y_pred = model.predict(np.array(x_test))
    to_save = list(map(lambda x: (x[0][0], x[0][1], x[1]), zip(test_rdd.collect(), y_pred)))
    write_csv(to_save, output_path)

    # y_pred = model.predict(np.array(x_train))
    # rmse = np.sqrt(mean_squared_error(y_pred, y_train))
    # print('in sample rmse is : ', rmse)

    end = time.time()
    print('finished in : ', end - start)