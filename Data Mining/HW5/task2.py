from random import randint
import binascii
import csv


def transform_id(x):
    return int(binascii.hexlify(x.encode('utf8')), 16)


def generate_hash_func(k):
    '''
    k: num of hash funcs
    '''
    hash_funcs = []
    for _ in range(k):
        def hash_func(x):
            '''
            input string id
            output bit array string
            '''
            out = ''
            x = transform_id(x)
            a = randint(1, 100)
            b = randint(1, 200)
            res = (a * x + b) % (2 ** 10)

            return res

        hash_funcs.append(hash_func)
    return hash_funcs




def myhashs(s):
    result = []
    for f in hash_func_list:
        result.append(f(s))
    return result


def split(lst, n=5):
    '''
    splitting into n chunks
    '''
    size = len(lst) // n
    res = []
    for i in range(n):
        res.append(lst[i * size: (i + 1) * size])
    return res


def fm(batch):
#     hash_func_list = generate_hash_func(k, n)
    res = []
    for x in batch:
        hashed = myhashs(x)
        hashed = [bin(x)[2:] for x in hashed]
        temp = [len(x.split('1')[-1]) for x in hashed]
        res.append(temp)
    Rs = [0] * len(res[0])
    for j in range(len(res[0])):
        Rs[j] = (max([res[i][j] for i in range(len(res))]))
    vals = [2**R for R in Rs]
    splitted_vals = split(vals, chunks)
    splitted_vals = [sum(x)/len(x) for x in splitted_vals]
    splitted_vals.sort()
    mid = chunks //2
    median = splitted_vals[mid]
    if chunks % 2 == 0:
        median = (median + splitted_vals[mid-1]) /2
    return median

def export_file(res, output_path):
    with open(output_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Time', 'Ground Truth', 'Estimation'])
        for pair in res:
            csv_writer.writerow(pair)
        print('writing finished')

if __name__ == '__main__':
    import sys
    import time
    from blackbox import BlackBox
    start = time.time()
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_ask = int(sys.argv[3])
    output_path = sys.argv[4]

    k = 30
    chunks = 10
    bx = BlackBox()
    res = []

    hash_func_list = generate_hash_func(k)
    for _ in range(num_ask):
        stm = bx.ask(file_name, stream_size)
        real = len(set(stm))
        pred = fm(stm)
        res.append((_, real, int(pred)))

    export_file(res, output_path)
    end = time.time()
    print(f'finished within {end - start} seconds! ')

