import sys
import timeit

import numpy as np
import multiprocessing
from multiprocessing import Pool

# split data for each workers
def partition_data(items, workers = 2):
    number_to_split = int(len(items) / workers)

    for i in range(workers + 1):
        yield(items[i * number_to_split:(i + 1) * (number_to_split)])

    remaining_group = items[(i + 1) * number_to_split:]
    if remaining_group:
        yield(remaining_group)


def map_function(numbers):

    print multiprocessing.current_process().name, 'map', numbers

    result = {}
    for i in numbers:
        try:
            result[i] += 1
        except KeyError:
            result[i] = 1
    return result


def reduce_function(sub_mapping):

    print multiprocessing.current_process().name, 'reduce', sub_mapping

    result = {}
    for i in sub_mapping:
        for k, v in i.items():
            try:
                result[k] += v
            except KeyError:
                result[k] = v
    return result


def mapreduce(all_items, map_workers, reduce_workers):

    group_items = list(partition_data(all_items, map_workers))
    pool = Pool(processes=map_workers)
    sub_map_result = pool.map(map_function, group_items)

    reduce_group_items = list(partition_data(sub_map_result, reduce_workers))
    pool2 = Pool(processes=reduce_workers)
    subresult = pool2.map(reduce_function, reduce_group_items)

    return reduce_function(subresult)


def show_result(dict):
    print "\nResult:"
    sorted_items = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for i in sorted_items:
        print("'%s' - %i" % i)


if __name__ == '__main__':

    map_workers = 4
    reduce_workers = 2

    # generate data
    data = np.random.randint(10, size=2048)

    final_result = mapreduce(data, map_workers, reduce_workers)

    show_result(final_result)
