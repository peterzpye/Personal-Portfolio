from itertools import combinations
from collections import defaultdict


from pyspark import SparkContext, SparkConf
conf = SparkConf().set("spark.executor.memory", "8g").set("spark.driver.memory", "8g")
sc = SparkContext.getOrCreate(conf = conf)

sc.setLogLevel("WARN")


def hash_func(a, b, val, n):
    return (((a*val) + b)%233333)%n

def min_hashing(rdd):
    bid, sets = rdd[0], rdd[1]
    vec = [max(all_users_info.values()) + 10] * hash_num
    for i, uid in enumerate(sets):
        val = all_users_info[uid]
        for m in range(hash_num):
            #creating different hashing seed
            a, b = m*3, m + 12
            hashed_val = hash_func(a,b,val, len(all_users_info)*3)
            vec[m] = min(hashed_val, vec[m])
    return (bid, vec)

def jaccard_similarity(a, b):
    '''
    array a and array b
    '''
    return len(set(a).intersection(set(b)))/len(set(a).union(set(b)))

def LSH(rdd):
    res = []
    r = list(rdd)
    buckets = defaultdict(list)
    min_r, max_r = min(r), max(r)+1
    for i in range(len(all_items)):
        item1 = all_items[i]
        string_version = ''.join([str(x) for x in minhash[item1][min_r: max_r]])
        bucket_num = hash(string_version)
        buckets[bucket_num].append(item1)
    yield list(buckets.items())



if __name__ == '__main__':
    import sys
    import time
    import csv
    import gc

    start = time.time()
    fpath, output_path = sys.argv[1], sys.argv[2]


    bins = 50
    r = 4
    hash_num = bins * r

    rdd = sc.textFile(fpath).filter(lambda x: x != 'user_id,business_id,stars') \
    .map(lambda x: (x.split(',')[1], [x.split(',')[0]] )) \
    .reduceByKey(lambda a,b: a + b).persist()

    #shingling
    all_items = rdd.map(lambda x: x[0]).distinct().collect()
    all_users = sc.textFile(fpath).filter(lambda x: x != 'user_id,business_id,stars') \
    .map(lambda x:  x.split(',')[0]).distinct().collect()
    all_items_info = {}
    for i, item in enumerate(all_items):
        all_items_info[item] = i
    all_users_info = {}
    for i, user in enumerate(all_users):
        all_users_info[user] = i
    item_info = rdd.collectAsMap()

    # min hashing
    # hash_func_list = get_hashfunc(hash_num, )
    minhash = rdd.map(lambda x: min_hashing(x)).collectAsMap()

    del all_items_info
    del all_users_info
    gc.collect()

    #LSH
    res = sc.parallelize(list(range(hash_num)), bins).mapPartitions(LSH) \
        .flatMap(lambda x: x) \
        .filter(lambda x: len(x[1]) >= 2) \
        .map(lambda x: x[1]) \
        .flatMap(lambda x: combinations(x, 2)) \
        .repartition(20) \
        .map(lambda x: ((tuple(sorted((x[0], x[1])))), jaccard_similarity(item_info[x[0]], item_info[x[1]]))) \
        .filter(lambda x: x[1] >= 0.5) \
        .distinct() \
        .sortByKey() \
        .map(lambda x: (x[0][0], x[0][1], x[1] )) \
        .collect()

    with open(output_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['business_id1','business_id2', 'similarity'])
        for pair in res:
            csv_writer.writerow(pair)
    end= time.time()
    print(f'finished in {end-start} seconds')

    ground_truth_path = 'C:/Users/yzp60/Downloads/DSCI553/HW3/pure_jaccard_similarity.csv'
    ground_truth_rdd = sc.textFile(ground_truth_path).filter(lambda x: x != 'business_id_1, business_id_2, similarity') \
        .map(lambda x: ((x.split(',')[0], x.split(',')[1]), 1))
    pred_info = sc.parallelize(res).map(lambda x: ((x[0], x[1]), 1)).collectAsMap()
    print('recall: ', ground_truth_rdd.map(lambda x: (x[0], x[1], \
                        pred_info.get((x[0][0], x[0][1]),pred_info.get((x[0][1], x[0][0]), 0)))) \
.map(lambda x: x[-1]).mean())

