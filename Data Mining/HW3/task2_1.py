import numpy as np

from pyspark import SparkContext, SparkConf
conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf = conf)
sc.setLogLevel("WARN")

def pearson_similarity(a, b, mode='corate'):
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


def recommend(x, rater_info, user_rating_info, user_avg_info, N = 5, item_threshold= 3, user_threshold = 1):
    tho = 2.5
    uid, bid = x[0], x[1]
    # score = x[2]
    try:
        items = user_rating_info[uid]
        cands = []
        for item in items:
            if item != bid:
                common_users = list(rater_info[item].intersection(rater_info[bid]))
                if len(common_users) >= user_threshold:
                    a = [score_info[(user, item)] for user in common_users]
                    b = [score_info[(user, bid)] for user in common_users]
                    sim = pearson_similarity(a ,b)
                    u_score = score_info[(uid, item)]
                    # case amplification
                    sim = sim * abs(sim )**(tho - 1)
                    cands.append((item, sim, u_score))


        if N is not None:
            cands.sort(key = lambda x :x[1])
            cands = cands[:N]
        if len(cands) >= item_threshold and np.mean([cand[1] for cand in cands]) > 0.2:
            pred = sum([cands[i][1] * cands[i][2] for i in range(len(cands))]) / sum \
                ([abs(cands[i][1]) for i in range(len(cands))])
            typ = 'sim_score'
        else:
            # new items:
            pred = user_avg_info[uid]
            typ = 'user_avg'

    except:
        # new users old item
        try:
            users = rater_info[bid]
            pred = np.mean([score_info[(user, bid)] for user in users])
            typ = 'item_avg'
        except:
            # new users new item
            pred = over_all_mean
            typ = 'overall_avg'
    # negative ratings
    if not pred > 0:
        users = rater_info[bid]
        pred = np.mean([score_info[(user, bid)] for user in users])
        typ = 'item_avg'
    # return (uid ,bid ,pred ,score ,typ)
    return (uid ,bid ,pred)


if __name__ == '__main__':
    import sys
    import time
    import csv
    import json
    start = time.time()
    fpath, test_fpath, output_file_path = sys.argv[1], sys.argv[2], sys.argv[3]

    N = 5
    threshold = 3

    train_rdd = sc.textFile(fpath).filter(lambda x: x != 'user_id,business_id,stars') \
        .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))
    test_rdd = sc.textFile(test_fpath, 6).filter(lambda x: x != 'user_id,business_id,stars') \
        .map(lambda x: (x.split(',')[0], x.split(',')[1]))

    train_rdd = train_rdd.persist()
    rater_info = train_rdd.map(lambda x: (x[1], set([x[0]]))).reduceByKey(lambda a, b: a.union(b)).collectAsMap()
    user_avg_info = train_rdd.map(lambda x: (x[0], [x[2]])).reduceByKey(lambda a, b: a + b).map(
        lambda x: (x[0], np.mean(x[1]))).collectAsMap()
    user_rating_info = train_rdd.map(lambda x: (x[0], set([x[1]]))).reduceByKey(lambda a, b: a.union(b)).collectAsMap()
    score_info = train_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
    over_all_mean = train_rdd.map(lambda x: x[2]).mean()
    res = test_rdd.map(lambda x: recommend(x, rater_info, user_rating_info, user_avg_info)).collect()

    #save outputs
    with open(output_file_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['user_id','business_id', 'prediction'])
        csv_writer.writerows(res)
    end = time.time()
    duration = end - start
    print(f'finished within {duration} seconds! ')

