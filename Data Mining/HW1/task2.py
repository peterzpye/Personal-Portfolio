from operator import add
import json
import time
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
sc.setLogLevel("WARN")


def task2(n_partition, tr):
    res = {}
    #default
    start = time.time()
    df = tr
    n_items_default = df.glom().map(len).collect()
    n_partition_default = df.getNumPartitions()
    business_review = tr.map(lambda record: [record['business_id'],1]).reduceByKey(add).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    end = time.time()
    exe_time = end - start
    default = {'n_partition': n_partition_default,
              'n_items':n_items_default,
              'exe_time':exe_time}

    #customized
    start = time.time()
    a = tr.map(lambda x: (x['business_id'], 1)).partitionBy(n_partition, lambda id:ord(id[0]))
    n_items = a.glom().map(len).collect()
    n_partition = a.getNumPartitions()
    ans = a.reduceByKey(add).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    end = time.time()
    exe_time = end - start
    customized = {
        'n_partition': n_partition,
          'n_items':n_items,
          'exe_time':exe_time
    }

    res['default'] = default
    res['customized'] = customized
    return res

if __name__ == '__main__':
    import sys

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_partition = int(sys.argv[3])
    data = sc.textFile(input_path).map(lambda row: json.loads(row))
    res = task2(n_partition, data)
    with open(output_path, 'w+') as output_file:
        json.dump(res, output_file)
    print('finished!')