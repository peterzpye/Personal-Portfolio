
import json
import time
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
sc.setLogLevel("WARN")

def task3A():
    start = time.time()
    dataset = review.map(lambda x: (x['business_id'], x['stars'])).join(business.map(lambda x: (x['business_id'], x['city'])))
    group_mean = dataset.map(lambda x: (x[1][1], (x[1][0],1))).reduceByKey(lambda a,b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0]/x[1]).collect()
    group_mean = sorted(group_mean, key = lambda x: (-x[1], x[0]))
    end = time.time()
    m1_exe_time = end - start
    return group_mean, m1_exe_time

def task3B():
    start = time.time()
    dataset = review.map(lambda x: (x['business_id'], x['stars'])).join(business.map(lambda x: (x['business_id'], x['city'])))
    group_mean = dataset.map(lambda x: (x[1][1], (x[1][0],1))).reduceByKey(lambda a,b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0]/x[1]).takeOrdered(10, key = lambda x: (-x[1], x[0]))
    end = time.time()
    m2_exe_time = end - start
    return m2_exe_time


if __name__ == '__main__':
    import sys

    review_path = sys.argv[1]
    business_path = sys.argv[2]
    output_path1 = sys.argv[3]
    output_path2 = sys.argv[4]
    review = sc.textFile(review_path).map(lambda row: json.loads(row))
    business = sc.textFile(business_path).map(lambda row: json.loads(row))
    output1,  m1_exe_time = task3A()
    m2_exe_time = task3B()
    res = {}
    res['m1'] = m1_exe_time
    res['m2'] = m2_exe_time
    op1 = open(output_path1, 'w+', encoding='utf-8')
    op1.writelines('city,stars\n')
    op1.writelines([x[0] + ',' + str(x[1]) + '\n' for x in output1])
    op1.close()
    print('write finished')

    with open(output_path2, 'w+') as op2:
        json.dump(res, op2)
    print('finished!')