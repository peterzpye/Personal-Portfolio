
from itertools import combinations
from pyspark import SparkContext, SparkConf
configuration = SparkConf()
configuration.set("spark.driver.memory", "4g")
configuration.set("spark.executor.memory", "4g")
sc = SparkContext.getOrCreate(configuration)
sc.setLogLevel("WARN")

# pcy


def check_inside(cand, vals):
    return set(cand).issubset(set(vals))

def shrink(record, frequent_singleton):
    """
    decreasing the size of the record to avoid low threshold issues
    """
    return list(set(record).intersection(set(frequent_singleton)))


def calculate_supports(cands, record, singleton):
    support = defaultdict(int)
    for cand in cands:
        for x in record:
            if check_inside(cand, x):
                support[cand] += 1
    return support

def make_pairs(cands, pair_num):
    '''
    cands: list of list of elements
    if single values, please store in [[1], [2],...]
    '''
    all_candidates = set()
    for cand in cands:
        for element in cand:
            all_candidates.add(element)
    all_candidates = list(all_candidates)
    new_pairs = combinations(all_candidates, pair_num)
    return new_pairs

def PCY_hash(a,b):
    return (int(a) + int(b)) %10

def pcy(data, threshold):
    '''
    data: a list of tuples (user_id,  its reviews), an iterator in RDD
    '''
    # convert to dictionary
    data = [x[1] for x in data]

    # first pass
    bitmap = defaultdict(int)
    supports = defaultdict(int)
    for x in data:
        for cand in x:
            supports[cand] += 1
        for pair in make_pairs([[elem] for elem in x], 2):
            bitmap[PCY_hash(pair[0], pair[1])] += 1

    record = data
    singleton = [tuple([x]) for x in supports.keys() if supports[x] >= threshold]

    all_pairs = list(singleton)
    # second pass
    doubleton = [x for x in make_pairs(singleton,2) if bitmap[PCY_hash(x[0], x[1])] >= threshold]
    supports = calculate_supports(doubleton, record, singleton)
    doubleton = [x for x in supports.keys() if supports[x] >= threshold]
    for cand in doubleton:
        all_pairs.append(cand)

    cands = doubleton
    pair_nums = 3
    while len(cands) > 1:
        cands = make_pairs(cands, pair_nums)
        supports = calculate_supports(cands, record, singleton)
        cands = [x for x in supports.keys() if supports[x] >= threshold]

        for cand in cands:
            all_pairs.append(cand)
        pair_nums += 1

    return all_pairs


def pcy_spark(iterator):
    data = [x for x in iterator]
    T = max(threshold*(len(data) /total_length), 1)
    # T = math.ceil(threshold * (len(data) / total_length))
    all_pairs = pcy(data, T)
    yield all_pairs


def SON_check(iterator, candidates):
    data = [x for x in iterator]
    for record in data:
        for cand in candidates:
            if check_inside(cand, record[1]):
                yield (cand, 1)

def to_print_format(frequent_items):
        length_count = defaultdict(list)
        for x in frequent_items:
            length_count[len(x)].append(x)
        frequent_items_print = []
        for pair_size in sorted(length_count.keys()):
            pairs = length_count[pair_size]
            frequent_items_print_subset = []
            # pairs = [sorted(pair) for pair in pairs]
            # pairs.sort()
            for pair in pairs:
                item = '('
                pair = sorted([str(int(x)) for x in pair])
                for i,x in enumerate(list(pair)):
                    item += "'" + str(x) + "'"
                    if i != len(pair)-1:
                        item += ', '
                item += ')'
                frequent_items_print_subset.append(item)
            frequent_items_print_subset.sort()
            frequent_items_print.append(frequent_items_print_subset)

        return frequent_items_print



if __name__ == '__main__':
    import time
    import sys
    from collections import defaultdict

    filter_threshold = int(sys.argv[1])
    threshold = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    partition_number = 8
    start = time.time()
    rdd = sc.textFile(input_path, partition_number) \
        .filter(lambda x: x.split(',')[0] != '"TRANSACTION_DT"') \
        .map(lambda line: (line.split(',')[0] + line.split(',')[1], str(int(line.split(',')[5].strip('"'))))) \
        .groupByKey().mapValues(set).reduceByKey(lambda a, b: a.union(b)).filter(lambda x: len(x[1]) > filter_threshold)
    total_length = rdd.count()

    # first pass
    cands = rdd.mapPartitions(pcy_spark).flatMap(lambda pair: pair).distinct().sortBy(
        lambda x: (len(x), x[0])).collect()
    cands = list(set(cands))
    intermediate = to_print_format(cands)
    middle = time.time()
    print('first pass finished in : ', middle - start)
    # second pass
    frequent_items = rdd.mapPartitions(lambda x: SON_check(x, cands)).reduceByKey(lambda a, b: a + b).filter(
        lambda x: x[1] >= threshold).sortBy(lambda x: len(x[0])).collect()
    frequent_items = [x[0] for x in frequent_items]
    res = to_print_format(frequent_items)

    with open(output_path, 'w+') as output_file:
        output_file.write('Candidates: \n')
        for pairs in intermediate:
            for i,x in enumerate(pairs):
                output_file.write(str(x))
                if i < len(pairs) - 1:
                    output_file.write(',')

            output_file.write('\n\n')
        output_file.write('Frequent Itemsets: \n')
        for pairs in res:
            for i, x in enumerate(pairs):
                output_file.write(str(x))
                if i < len(pairs) - 1:
                    output_file.write(',')
            output_file.write('\n\n')
    end = time.time()
    duration = end - start

    print('finished with duration: ', duration)