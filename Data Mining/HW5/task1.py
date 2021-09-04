from random import randint
import binascii
from copy import deepcopy
import csv


def generate_hash_func(n, m):
    hash_funcs = []
    for _ in range(n):
        def hash_func(x):
            '''
            input string id
            output hashed position
            '''
            x = transform_id(x)
            a = randint(1, 100)
            b = randint(1, 200)
            res = ((a * x + b) % 23333333) % m
            return res

        hash_funcs.append(hash_func)
    return hash_funcs


def transform_id(x):
    return int(binascii.hexlify(x.encode('utf8')), 16)


def myhashs(s):
    res = []
    for f in hash_func_lst:
        res.append(f(s))
    return res


def bloom_filter(x_lst, arr_dict):
    res = {}

    arr_updated = deepcopy(arr_dict)
    for x in x_lst:
        new = False
        positions = myhashs(x)
        for pos in positions:
            if arr_dict[pos] == 0:
                arr_updated[pos] = 1
                new = True
        if new == True:
            res[x] = 0
        else:
            res[x] = 1
    return res, arr_updated

def export_file(res, output_path):
    with open(output_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Time', 'FPR'])
        for pair in res:
            csv_writer.writerow(pair)
        print('writing finished')


if __name__ == '__main__':
    from blackbox import BlackBox
    import sys
    import time


    start = time.time()
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_ask = int(sys.argv[3])
    output_path = sys.argv[4]

    m = 69997
    n = 2

    bx = BlackBox()
    arr_dict = [0] * m
    truth = set()
    record = []
    for _ in range(num_ask):

        stream_users = bx.ask(file_name, stream_size)
        hash_func_lst = generate_hash_func(n, m)
        res, arr_dict = bloom_filter(stream_users, arr_dict)
        fpr = []

        for uid in res:
            if res[uid] == 1 and uid not in truth:
                fpr.append(1)
            fpr.append(0)

        fpr_rate = sum(fpr) / len(fpr)
        record.append((_, fpr_rate))
        for uid in stream_users:
            truth.add(uid)
    export_file(record, output_path)
    end = time.time()
    print(f'finished within {end - start} seconds! ')