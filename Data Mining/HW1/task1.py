
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
sc.setLogLevel("WARN")


def task1(tr):
    from operator import add
    import time

    start = time.time()


    ans = {}


    # Q1
    num_review = tr.count()
    ans['n_review'] = num_review

    # Q2
    def q2(data):
        if data['date'].split('-')[0] == '2018':
            return 1
        return 0
    review_2018 = tr.map(q2).reduce(lambda a,b:a+b)
    ans['n_review_2018'] = review_2018

    #Q3
    distinct_user = tr.map(lambda record: (record['user_id'],1)).reduceByKey(lambda a,b: 1).count()
    ans['n_user'] = distinct_user

    #Q4
    user_review = tr.map(lambda record: [record['user_id'],1]).reduceByKey(add).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    ans['top10_user'] = [list(x) for x in user_review]

    #Q5
    distinct_business = tr.map(lambda record: (record['business_id'],1)).reduceByKey(lambda a,b: 1).count()
    ans['n_business'] = distinct_business

    #Q6
    business_review = tr.map(lambda record: [record['business_id'],1]).reduceByKey(add).takeOrdered(10, key=lambda x: (-x[1], x[0]))
    ans['top10_business'] = [list(x) for x in business_review]


    end = time.time()
    print(f'total elapsed time: {end - start}s')
    return ans

if __name__ == '__main__':
    from operator import add
    import json
    import time
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    data = sc.textFile(input_path).map(lambda row: json.loads(row))
    res = task1(data)
    with open(output_path, 'w+') as output_file:
        json.dump(res, output_file)
    print('finished!')