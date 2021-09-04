from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local[*]").set('spark.executor.memory', '4g').set('spark.driver.memory', '4g')
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("WARN")


from collections import defaultdict
import math
import numpy as np
from numpy.random import shuffle
from sklearn.cluster import KMeans
from copy import deepcopy

def split(data, n):
    '''
    split data into n chunks
    '''
    shuffle(data)
    chunk_size = len(data)//n

    splitted = []
    i = 0
    start = 0
    for i in range(n):
        dat = data[i*chunk_size:(i+1)*chunk_size]
        splitted.append(dat)
    remaining = data[n*chunk_size:]
    for i in range(len(remaining)):
        splitted[i].append(remaining[i])

    return splitted

def random_hash(idx):
    '''
    implementation of random choice of 20% data,
    hash the index of data points
    '''
    return ((13333 * idx + 3)%2333333)%5

def mahalanobis_distance(N, centroid_sum , centroid_sum_sq, point):
    '''
    centroid: vec
    points: vec
    '''
    point = np.array(point)
    centroid_sum = np.array(centroid_sum)
    centroid_sum_sq = np.array(centroid_sum_sq)

    sigma = np.sqrt(centroid_sum_sq/N - (centroid_sum/N)**2)
    distance = np.sqrt((((point - centroid_sum/N)/sigma)**2).sum())
    return distance


def initialize(chunk_data_info, DS, RS, CS):
    # first pass
    km = KMeans(n_clusters=8 * N)
    km.fit(list(chunk_data_info.values()))

    init_info = defaultdict(list)

    for i in range(len(chunk_data_info)):
        init_info[km.labels_[i]].append(list(chunk_data_info.keys())[i])

    for label in init_info:
        if len(init_info[label]) == 1:
            elem = init_info[label][0]
            RS[elem] = chunk_data_info[elem]
            del chunk_data_info[elem]
    # second pass
    km = KMeans(n_clusters=N)
    km.fit(list(chunk_data_info.values()))
    DS, RS = extract_set(chunk_data_info, km.labels_, DS, RS)
    return DS, RS, CS


def extract_set(chunk_data_info, km_labels, add_set, RS, replace_RS=False):
    '''
    using the label, put clusters into add_set, else put into RS
    '''
    chunk_data_info = deepcopy(chunk_data_info)
    new_RS = {}
    cluster_info = defaultdict(list)
    for i in range(len(chunk_data_info)):
        cluster_info[km_labels[i]].append(list(chunk_data_info.keys())[i])  # contains a list of candidates idx
    for label in cluster_info:
        if len(cluster_info[label]) > 1:
            elem = np.array([chunk_data_info[x] for x in cluster_info[label]])  # feature matrix
            cands = cluster_info[label]
            length = len(elem)
            sum_vec = elem.sum(axis=0)
            sum_sq_vec = (elem ** 2).sum(axis=0)
            add_set[label] = (length, sum_vec, sum_sq_vec, cands)
        else:
            cand = cluster_info[label][0]
            RS[cand] = chunk_data_info[cand]
            new_RS[cand] = chunk_data_info[cand]
    if replace_RS == False:
        return add_set, RS
    return add_set, new_RS


def cluster_RS(RS, CS):
    #     N = self.N
    RS, CS = deepcopy(RS), deepcopy(CS)
    if len(RS) > 3 * N:
        RS_mt = list(RS.values())
        km = KMeans(n_clusters=3 * N)
        km.fit(RS_mt)
        labels = km.labels_
        CS, RS = extract_set(RS, labels, CS, RS, replace_RS=True)

        return RS, CS
    else:

        return RS, CS


def merge_CS(CS):
    CS = deepcopy(CS)
    to_delete = set()
    for cs_cluster in CS:
        closest = None
        lowest_distance = math.inf
        length, sum_vec, sum_sq_vec, cands = CS[cs_cluster]
        vec = sum_vec / length
        for cs_cluster2 in CS:
            if cs_cluster2 != cs_cluster and cs_cluster2 not in to_delete:
                distance = mahalanobis_distance(CS[cs_cluster2][0], CS[cs_cluster2][1], CS[cs_cluster2][2], vec)
                if distance < lowest_distance:
                    closest = cs_cluster2
                    lowest_distance = distance
        if lowest_distance < 2 * np.sqrt(len(vec)):
            length_or, sum_vec_or, sum_sq_vec_or, cands_or = CS[closest]
            CS[closest] = (length + length_or, sum_vec + sum_vec_or, sum_sq_vec + sum_sq_vec_or, cands + cands_or)
            to_delete.add(cs_cluster)
    for cluster in to_delete:
        del CS[cluster]
    return CS


def one_pass(chunk_data_info, DS, RS, CS):
    for label in chunk_data_info:
        DS_point, CS_point = False, False
        vec = np.array(chunk_data_info[label])
        vec_sq = vec ** 2
        closest = None
        lowest_distance = math.inf
        #
        for ds_cluster in DS:
            distance = mahalanobis_distance(DS[ds_cluster][0], DS[ds_cluster][1], DS[ds_cluster][2], vec)
            if distance < lowest_distance:
                closest = ds_cluster
                lowest_distance = distance
        if lowest_distance < 2 * np.sqrt(len(vec)):
            length_or, sum_vec_or, sum_sq_vec_or, cands_or = DS[closest]
            DS[closest] = (1 + length_or, vec + sum_vec_or, vec_sq + sum_sq_vec_or, [label] + cands_or)
            DS_point = True
        if DS_point == False:
            closest = None
            lowest_distance = math.inf
            for cs_cluster in CS:
                distance = mahalanobis_distance(CS[cs_cluster][0], CS[cs_cluster][1], CS[cs_cluster][2], vec)
                if distance < lowest_distance:
                    closest = cs_cluster
                    lowest_distance = distance
            if lowest_distance < 2 * np.sqrt(len(vec)):
                length_or, sum_vec_or, sum_sq_vec_or, cands_or = CS[closest]
                CS[closest] = (1 + length_or, vec + sum_vec_or, vec_sq + sum_sq_vec_or, [label] + cands_or)
                CS_point = True
        if DS_point == False and CS_point == False:
            RS[label] = chunk_data_info[label]

    RS, CS = cluster_RS(RS, CS)

    CS = merge_CS(CS)

    return DS, RS, CS


def merge_DSCS(DS, CS):
    to_delete = []
    for cs_cluster in CS:
        closest = None
        lowest_distance = math.inf
        length, sum_vec, sum_sq_vec, cands = CS[cs_cluster]
        vec = sum_vec / length
        for ds_cluster in DS:
            distance = mahalanobis_distance(DS[ds_cluster][0], DS[ds_cluster][1], DS[ds_cluster][2], vec)
            if distance < lowest_distance:
                closest = ds_cluster
                lowest_distance = distance
        if lowest_distance < 2 * np.sqrt(len(vec)):
            length_or, sum_vec_or, sum_sq_vec_or, cands_or = DS[closest]
            DS[closest] = (length + length_or, sum_vec + sum_vec_or, sum_sq_vec + sum_sq_vec_or, cands + cands_or)
            to_delete.append(cs_cluster)  # append cluster numbers to delete

    for c in to_delete:
        del CS[c]
    return DS, CS

def export_res(outpath, DS, CS, RS, record_history):
    res = []
    for label in DS:
        for datapoint in DS[label][-1]:
            res.append((datapoint, label))
    for label in CS:
        for datapoint in CS[label][-1]:
            res.append((datapoint, -1))
    for label in RS:
        res.append((label, -1))
    with open(outpath, 'w+') as f:
        f.write('The intermediate results:\n')
        for i in range(len(record_history)):
            x = record_history[i]
            content = f'Round {i+1}: {x[0]},{x[1]},{x[2]},{x[3]}\n'
            f.write(content)
        f.write('\n')
        f.write('The clustering results:\n')
        for x in sorted(res, key = lambda x: x[0]):
            content = f'{x[0]},{x[1]}\n'
            f.write(content)
    print('writing finished')


if __name__ == '__main__':
    import time
    import sys
    time1 = time.time()
    fpath = sys.argv[1]
    N = int(sys.argv[2])
    outpath = sys.argv[3]

    DS = {}
    RS = {}
    CS = {}
    record_history = []
    num_pass = 5
    discarded = 0

    data_rdd = sc.textFile(fpath) \
        .map(lambda x: (random_hash(int(x.split(',')[0])), (int(x.split(',')[0]), list(map(float, x.split(',')[2:]))))) \
        .persist()
    chunk_data_rdd = data_rdd.filter(lambda x: x[0] == 0).map(lambda x: x[1])
    chunk_data_info = chunk_data_rdd.collectAsMap()

    DS, RS, CS = initialize(chunk_data_info, DS, RS, CS)

    RS, CS = cluster_RS(RS, CS)
    discarded = sum([len(x[-1]) for x in DS.values()])
    compressed = sum([len(x[-1]) for x in CS.values()])
    n_rs = len(RS)
    n_cs = len(CS)
    round_info = (discarded, n_cs, compressed, n_rs)
    record_history.append(round_info)
    print(f'Round 1: {round_info}')
    print('started')

    for i in range(1, num_pass):
        start = time.time()
        chunk_data_rdd = data_rdd.filter(lambda x: x[0] == i).map(lambda x: x[1])
        chunk_data_info = chunk_data_rdd.collectAsMap()
        DS, RS, CS = one_pass(chunk_data_info, DS, RS, CS)
        if i == num_pass - 1:
            DS, CS = merge_DSCS(DS, CS)

        discarded = sum([len(x[-1]) for x in DS.values()])
        compressed = sum([len(x[-1]) for x in CS.values()])
        n_rs = len(RS)
        n_cs = len(CS)
        round_info = (discarded, n_cs, compressed, n_rs)
        record_history.append(round_info)
        end = time.time()
        print(f'Round {i + 1}: {round_info}')
        print(f'finished {i}th chunk within {end - start} seconds!')

    export_res(outpath, DS, CS, RS, record_history)
    time2 = time.time()
    print(f'finished in total {time2 - time1} seconds!')


