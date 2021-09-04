import random
import binascii
from copy import deepcopy
import csv



def export_file(res, output_path):
    with open(output_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])
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

    random.seed(553)
    bx = BlackBox()
    data_holder = []
    n = 0
    res = []
    for _ in range(num_ask):
        stream_users = bx.ask(file_name, stream_size)
        for user in stream_users:
            n += 1
            if len(data_holder) < 100:
                data_holder.append(user)
            else:
                prob = random.random()
                if prob < 100 / n:
                    idx = random.randint(0, 99)
                    data_holder[idx] = user
        res.append((n, data_holder[0], data_holder[20], data_holder[40], data_holder[60], data_holder[80]))
    export_file(res, output_path)
    end = time.time()
    print(f'finished within {end - start} seconds! ')