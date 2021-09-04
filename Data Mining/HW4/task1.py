from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql import functions as F

conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf = conf)
sc.setLogLevel("ERROR")
spark = SparkSession(sc)

def count_common(rdd):
    uid = rdd[0]
    res = []
    for idx in uid_lst:
        if idx != uid:
            infoa = set(bid_lst_info[idx])
            infob = set(bid_lst_info[uid])
            common = len(infoa.intersection(infob))
            if common >= filter_threshold:
                res.append(((tuple(sorted([uid, idx]))), common ))
    return res

def write_files(output_path, ans):
    ans = sorted(ans, key=lambda x: (len(x), x[0]))
    with open(output_path, 'w+') as f:
        for community in ans:
            community = ['"' + x + '"' for x in community]
            content = ', '.join(community)
            content += '\n'
            f.write(content)
    print('writing finished')


if __name__ == '__main__':
    import sys
    import time
    import os

    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

    filter_threshold = 7
    fpath = sys.argv[1]
    output_path = sys.argv[2]


    start = time.time()

    rdd = sc.textFile(fpath).filter(lambda x: x != 'user_id,business_id') \
        .map(lambda x: (x.split(',')[0], x.split(',')[1])) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))).persist()
    uid_lst = rdd.map(lambda x: x[0]).collect()
    bid_lst_info = rdd.collectAsMap()
    edges = rdd.flatMap(count_common).distinct().map(lambda x: (x[0][0], x[0][1])).toDF(["src", "dst"])
    v = rdd.map(lambda x: (x[0],)).toDF(['id'])

    gf = GraphFrame(v, edges).dropIsolatedVertices()
    result = gf.labelPropagation(maxIter=5)
    ans_rdd = result.rdd.coalesce(4).map(tuple)
    ans = ans_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list).mapValues(sorted).sortBy(
        lambda x: len(x[1])).map(lambda x: x[1]).collect()

    #export answers
    write_files(output_path, ans)

    end = time.time()
    duration = end - start
    print(f'finished in {duration} seconds!')