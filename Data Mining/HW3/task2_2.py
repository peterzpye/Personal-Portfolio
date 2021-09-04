from xgboost import XGBRegressor
import json
import numpy as np

from pyspark import SparkContext, SparkConf
conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf = conf)
sc.setLogLevel("WARN")


def train_test_split(x, y):
    N = len(x)
    ids = list(range(N))
    np.random.shuffle(ids)
    train_ids, test_ids = ids[:int(N*0.8)], ids[int(N*0.8):]
    return x[train_ids], y[train_ids], x[test_ids], y[test_ids]


def derive_business_features(row):
    try:
        alcohol = int(bool(row['attributes']['Alcohol']))
    except:
        alcohol = 0
    try:
        dogs = int(bool(row['attributes']['DogsAllowed']))
    except:
        dogs = 0
    try:
        cater = int(bool(row['attributes']['Caters']))
    except:
        cater = 0
    try:
        goodforkid = int(bool(row['attributes']['GoodForKids']))
    except:
        goodforkid = 0
    try:
        outdoor = int(bool(row['attributes']['OutdoorSeating']))
    except:
        outdoor = 0
    try:
        review_count = int(row['review_count'])
    except:
        review_count = 0
    try:
        star = int(row['stars'])
    except:
        star = 2.5
    try:
        categories = len(row['categories'].split(','))
    except:
        categories = 0
    return (row['business_id'],  \
                  (star,
                   review_count, \
                   categories, \
                   alcohol, \
                   outdoor \
                  ))


def calculate_item_info(rdd):
    uid, bid = rdd[0], rdd[1]
    users = rater_info[bid]
    res = []
    threshold = 3

    for user in users:
        if user != uid:
            res.append(score_info[(user, bid)])
    if len(res) > 0 and len(res) < threshold:
        ans = (np.mean([avg_item_avg, np.mean(res)]), np.mean([avg_item_std, np.mean(res)]))
    elif len(res) >= threshold:
        ans = (np.mean(res), np.std(res))
    else:
        ans = (avg_item_avg, avg_item_std)
    return ((uid, bid), ans)

def calculate_user_info(rdd):
    uid, bid = rdd[0], rdd[1]
    threshold = 3
    items = user_rating_info[uid]
    res = []
    for item in items:
        if item != bid:
            res.append(score_info[(uid, item)])
    if len(res) > 0 and len(res) < threshold:
        ans = (np.mean([avg_user_avg, np.mean(res)]), np.mean([avg_user_std, np.std(res)]))
    elif len(res) >= threshold:
        ans = (np.mean(res), np.std(res))
    else:
        ans = (avg_user_avg, avg_user_std)
    return ((uid, bid), ans)



def combine(rdd):
    uid, bid, score = rdd[0], rdd[1], rdd[2]
    num_raters = rater_count_info[bid]
    num_rating = user_count_info[uid]
    user_avg, user_std = user_avg_info[(uid, bid)]
    item_avg, item_std = item_avg_info[(uid, bid)]

    business_feat = business_feats.get(bid, business_feats_mean)
    checkin_feat = checkin_feats.get(bid, checkin_feats_mean)
    checkin_feat = tuple([checkin_feat])
    return ((uid, bid), ( score, num_raters, num_rating, user_avg, item_avg, user_std, item_std) + checkin_feat + business_feat )

def combine_test(rdd):
    '''
    make the feature space including filling NAN values
    '''
    uid, bid, score = rdd[0], rdd[1], rdd[2]
    try:
        num_raters = rater_count_info[bid]
    except:
        num_raters = avg_rater_count
    try:
        num_rating = user_count_info[uid]
    except:
        num_rating = avg_rating_count
    try:
        user_avg, user_std = user_avg_info_test[uid]
    except:
        user_avg, user_std = avg_user_avg, avg_user_std
    try:
        item_avg, item_std = item_avg_info_test[bid]
    except:
        item_avg, item_std = avg_item_avg, avg_item_std
#         item_avg = np.nan

    business_feat = business_feats.get(bid, business_feats_mean)
    checkin_feat = checkin_feats.get(bid, checkin_feats_mean)
    checkin_feat = tuple([checkin_feat])
    return ((uid, bid), (score, num_raters, num_rating, user_avg, item_avg, user_std, item_std) + checkin_feat + business_feat )

def combine_pred(rdd):
    '''
    make the feature space including filling NAN values
    '''
    uid, bid = rdd[0], rdd[1]
    try:
        num_raters = rater_count_info[bid]
    except:
        num_raters = avg_rater_count
    try:
        num_rating = user_count_info[uid]
    except:
        num_rating = avg_rating_count
    try:
        user_avg, user_std = user_avg_info_test[uid]
    except:
        user_avg, user_std = avg_user_avg, avg_user_std
    try:
        item_avg, item_std = item_avg_info_test[bid]
    except:
        item_avg, item_std = avg_item_avg, avg_item_std
    business_feat = business_feats.get(bid, business_feats_mean)
    checkin_feat = checkin_feats.get(bid, checkin_feats_mean)
    checkin_feat = tuple([checkin_feat])
    return ((uid, bid), ( num_raters, num_rating, user_avg, item_avg, user_std, item_std) + checkin_feat + business_feat )


if __name__ == '__main__':
    import sys
    import time
    import csv
    import json
    start = time.time()
    folder_path, test_fpath, output_file_path = sys.argv[1], sys.argv[2], sys.argv[3]
    business_path = folder_path + '/business.json'
    checkin_path = folder_path + '/checkin.json'
    fpath = folder_path + '/yelp_train.csv'


    train_rdd = sc.textFile(fpath).filter(lambda x: x != 'user_id,business_id,stars') \
        .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))
    test_rdd = sc.textFile(test_fpath).filter(lambda x: x != 'user_id,business_id,stars') \
        .map(lambda x: (x.split(',')[0], x.split(',')[1]))

    business_feats = sc.textFile(business_path).map(lambda row: json.loads(row)) \
        .map(lambda row: derive_business_features(row)) \
        .collectAsMap()
    checkin_feats = sc.textFile(checkin_path).map(lambda row: json.loads(row)) \
        .map(lambda x: (x['business_id'], np.sum(list(x['time'].values())))) \
        .reduceByKey(lambda a, b: a + b) \
        .collectAsMap()

    train_rdd = train_rdd.persist()
    rater_info = train_rdd.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda a, b: a.union(b)).collectAsMap()
    rater_count_info = train_rdd.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda a, b: a.union(b)).map(
        lambda x: (x[0], len(x[1]))).collectAsMap()
    user_count_info = train_rdd.map(lambda x: (x[0], set([x[1]]))).reduceByKey(lambda a, b: a.union(b)).map(
        lambda x: (x[0], len(x[1]))).collectAsMap()
    user_rating_info = train_rdd.map(lambda x: (x[0], set([x[1]]))).reduceByKey(lambda a, b: a.union(b)).collectAsMap()
    score_info = train_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    over_all_mean = train_rdd.map(lambda x: x[2]).mean()
    business_feats_mean = tuple([np.mean(list(business_feats.values()), axis=0)[0]] + [0 for _ in range(
        (len(list(business_feats.values())[0]) - 1))])
    checkin_feats_mean = np.mean(np.array(list(checkin_feats.values())))
    avg_user_avg = train_rdd.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: np.mean(x[1])).mean()
    avg_item_avg = train_rdd.map(lambda x: (x[1], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: np.mean(x[1])).mean()
    avg_user_std = train_rdd.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: np.std(x[1])).mean()
    avg_item_std = train_rdd.map(lambda x: (x[1], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: np.std(x[1])).mean()

    user_avg_info = train_rdd.map(lambda x: calculate_user_info(x)).collectAsMap()
    item_avg_info = train_rdd.map(lambda x: calculate_item_info(x)).collectAsMap()

    avg_rater_count = np.mean(list(rater_count_info.values()))
    avg_rating_count = np.mean(list(user_count_info.values()))

    user_avg_info_test = train_rdd.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: (x[0], (np.mean(x[1]), np.std(x[1])))).collectAsMap()
    item_avg_info_test = train_rdd.map(lambda x: (x[1], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: (x[0], (np.mean(x[1]), np.std(x[1])))).collectAsMap()

    print('all featrues data loaded!')

    # training model
    data_info = train_rdd.map(lambda x: combine(x)).collectAsMap()
    data = np.array(list(data_info.values()))
    x, y = data[:, 1:], data[:, 0].reshape(-1, 1)
    print('training dataset ready! ')

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.025,
        reg_alpha=0.1,
        max_depth=5,
    )
    model.fit(x, y)
    print('model fitted')

    #making prediction
    pred_info = test_rdd.map(lambda x: combine_pred(x))
    res = pred_info.map(lambda x: (x[0][0], x[0][1], model.predict(np.array(x[1]).reshape(1,-1))[0])).collect()

    #save outputs
    with open(output_file_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['user_id','business_id', 'prediction'])
        for pair in res:
            csv_writer.writerow(pair)
    end = time.time()
    duration = end - start
    print(f'finished within {duration} seconds! ')


