

from pyspark import SparkContext, SparkConf
conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf = conf)
sc.setLogLevel("ERROR")


from collections import defaultdict
from copy import deepcopy

def count_common(rdd):
    uid = rdd[0]
    res = []
    for idx in uid_lst:
        if idx != uid:
            infoa = set(bid_lst_info[idx])
            infob = set(bid_lst_info[uid])
            common = len(infoa.intersection(infob))
            if common >= filter_threshold:
                res.append(((tuple(sorted([uid, idx]))), common))
    return res


def compute_betweeness(rdd, edge_info):
    node = rdd
    queue = [node]
    # how many parent links each node has
    links = defaultdict(int)
    # use bfs to traverse
    reversed_levels = []
    path = set()
    path.add(node)
    while queue:
        level = []
        for _ in range(len(queue)):

            parent = queue.pop(0)
            level.append(parent)
            children = edge_info.get(parent, [])

            for child in children:
                if child not in path:
                    queue.append(child)
                    path.add(child)
        reversed_levels.insert(0, level)
    # loop by levels to get the number of shortest path per node
    levels = reversed_levels[::-1]
    links[node] = 1
    for i in range(len(levels) - 1):
        for parent in levels[i]:
            for child in edge_info.get(parent, []):
                if child in set(levels[i + 1]):
                    links[child] += links[parent]
    # values of each node
    val = {x: 1 for x in links if x != node}
    val[node] = 0
    output = []
    for i in range(1, len(reversed_levels)):
        level = reversed_levels[i]
        prev_level = reversed_levels[i - 1]
        for parent in level:
            children = edge_info.get(parent, [])
            for child in children:
                if child in set(prev_level):
                    val[parent] += val[child] * links[parent] / links[child]
                    a, b = sorted([parent, child])
                    output.append(((a, b), val[child] * links[parent] / links[child]))

    return output


def find_neighbors(rdd, edge_info):
    '''
    rdd is the uid
    '''

    node = rdd
    path = set()
    queue = [node]
    while queue:
        parent = queue.pop(0)
        path.add(parent)
        children = edge_info.get(parent, [])
        for child in children:
            if child not in path:
                queue.append(child)

    return ','.join(sorted(list(path)))


def Compute_Modularity_Q(rdd):
    '''
    modularity_q for a single community, community comes in string format
    return a single score for this community
    '''
    community = rdd.split(',')
    res = 0
    for a in community:
        for b in community:
            A = Am.get(tuple([a, b]), Am.get(tuple([b, a]), 0))
            ka = degree_info[a]
            kb = degree_info[b]
            res += A - ka * kb / (2 * m)
    return res


def write_betweeness(output_path, betweeness):
    for i in range(len(betweeness)):
        pair = betweeness[i]
        ids, score = pair[0], pair[1]
        betweeness[i] = (tuple(sorted(ids)), score)
    betweeness = sorted(betweeness, key=lambda x: (-x[1], x[0][0]))
    with open(output_path, 'w+') as f:
        for pair in betweeness:
            ids, score = pair[0], pair[1]
            content = f"('{ids[0]}', '{ids[1]}'),{round(score,5)}\n"
            f.write(content)
    print('betweeness written successfully')


def write_files(output_path, ans):
    ans = sorted(ans, key=lambda x: (len(x), x[0]))
    with open(output_path, 'w+') as f:
        for community in ans:
            community.sort()
            community = ['"' + x + '"' for x in community]
            content = ', '.join(community)
            content += '\n'
            f.write(content)
    print('community written successfully')


if __name__ == '__main__':
    import sys
    import time


    filter_threshold = int(sys.argv[1])
    fpath = sys.argv[2]
    betweeness_output_path = sys.argv[3]
    community_output_path = sys.argv[4]

    start = time.time()

    max_iter = 1000
    EARLY_STOPPING = 30


    # initiate
    rdd = sc.textFile(fpath, 1).filter(lambda x: x != 'user_id,business_id') \
        .map(lambda x: (x.split(',')[0], x.split(',')[1])) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1])))

    # for count_common
    uid_lst = rdd.map(lambda x: x[0]).collect()
    bid_lst_info = rdd.collectAsMap()

    edge_rdd = rdd.flatMap(count_common).distinct().map(lambda x: (x[0][0], x[0][1])).coalesce(2)
    degree_info = edge_rdd.flatMap(lambda x: [(x[0], 1), (x[1], 1)]).groupByKey().mapValues(sum).collectAsMap()
    Am = edge_rdd.map(lambda x: (tuple(sorted(x)), 1)).collectAsMap()
    m = edge_rdd.count()
    v = list(set(degree_info.keys()))

    edge_info = edge_rdd.flatMap(lambda x: [(x[0], x[1]), (x[1], x[0])]).groupByKey().mapValues(set).collectAsMap()

    communities_rdd = sc.parallelize(v).map(lambda x: find_neighbors(x, edge_info)).distinct()
    Modularity_Q = communities_rdd.map(Compute_Modularity_Q).sum() / (2 * m)
    max_q = Modularity_Q
    tolerance = 0
    max_step = 0

    preprocess_end = time.time()
    print(f'preprocessing ended in {preprocess_end - start} seconds...')

    for step in range(max_iter):
        print('cutting started')
        round_start = time.time()
        betweeness = sc.parallelize(v, 1) \
            .flatMap(lambda x: compute_betweeness(x, edge_info)) \
            .reduceByKey(lambda a, b: a + b) \
            .mapValues(lambda x: x / 2) \
            .sortBy(lambda x: -x[1]) \
            .collect()
        if step == 0:
            write_betweeness(betweeness_output_path, betweeness)
        highest_betweeness = betweeness[0][1]
        to_cut = set()
        i = 0
        while betweeness[i][1] >= highest_betweeness:
            to_cut.add((betweeness[i][0]))

            i += 1
        for pair in to_cut:
            edge_info[pair[0]].discard(pair[1])
            edge_info[pair[1]].discard(pair[0])

        communities_rdd = sc.parallelize(v, 1).map(lambda x: find_neighbors(x, edge_info)).filter(
            lambda x: len(x) > 0).distinct()
        Modularity_Q = communities_rdd.map(Compute_Modularity_Q).sum() / (2 * m)

        if Modularity_Q >= max_q:
            max_q = Modularity_Q
            tolerance = 0
            res_edge_info = deepcopy(edge_info)
            max_step = step

        else:
            tolerance += 1
        #     if Modularity_Q< max_q:
        #         break
        if tolerance >= EARLY_STOPPING:
            break
        #     edge_rdd = edge_rdd.filter(lambda x: x not in to_cut)
        round_end = time.time()
        print(f'{step} done with q score: {Modularity_Q}, using {round_end - round_start} seconds')

    res = sc.parallelize(v, 1).map(lambda x: find_neighbors(x, res_edge_info)).filter(
        lambda x: len(x) > 0).distinct().map(lambda x: x.split(',')).collect()


    end = time.time()
    print(f'finished within {end - start} seconds! ')
    print(f'final q score is {max_q}, with {max_step} cut')

    write_files(community_output_path, res)

