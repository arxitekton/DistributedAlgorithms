import multiprocessing
import math

import numpy as np
import pandas as pd

import timeit

from collections import OrderedDict

n = 16
input_data = np.asarray([2**i for i in range(1, n + 1)])



def seq_prefixsum(input_data, n):
    prefix_sum_cpu = np.zeros((n + 1,), dtype=int)

    for i in range(1,n + 1):
        prefix_sum_cpu[i] = prefix_sum_cpu[i-1] + input_data[i-1]
    return prefix_sum_cpu


# Parallel
def summator(array, level, num, proc):

    lbound = proc * num
    rbound = proc * (num + 1)

    for sum_index in range(lbound + pow(2, level) - 1, rbound, pow(2, level)):
        array[sum_index] = array[sum_index] + array[sum_index - pow(2, (level - 1))]

    return


def down_summator(array, level, num, proc):

    lbound = proc * num
    rbound = proc * (num + 1)

    for sum_index in range(lbound + pow(2, level) - 1, rbound, pow(2, level)):
        val = array[sum_index]
        array[sum_index] = array[sum_index] + array[sum_index - pow(2, (level - 1))]
        array[sum_index - pow(2, (level - 1))] = val

    return


def multiSumming(jobs, array, length, level, asum_func, max_cores):
    processing_field = pow(2, level)
    core_number = int(length / processing_field)

    if core_number > max_cores:
        core_number = max_cores
        processing_field = int(length / core_number)

    for i in range(core_number):
        p = multiprocessing.Process(target=asum_func, args=(array, level, i, processing_field))
        p.daemon = False
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    return


def multiproc_prefixsum(input_data, n):

    max_cores = multiprocessing.cpu_count()

    length = n

    lst = input_data.copy()
    lst = np.append(lst, 0)

    lst[length] = 0

    array = multiprocessing.Array("i", lst)

    jobs = []
    depth = int(math.log(length, 2))

    for level in range(1, int(depth) + 1):
        multiSumming(jobs, array, length, level, summator, max_cores)

    cumsum = array[length - 1]

    array[length - 1] = 0
    core_number = 1

    for level in range(int(depth), 0, -1):
        multiSumming(jobs, array, length, level, down_summator, max_cores)
        array[length] = cumsum

    return np.asarray(array[:])


# some test
print input_data
print multiproc_prefixsum(input_data, n)
print seq_prefixsum(input_data, n)

#[    2     4     8    16    32    64   128   256   512  1024  2048  4096  8192 16384 32768 65536]
#[     0      2      6     14     30     62    126    254    510   1022   2046   4094   8190  16382  32766  65534 131070]
#[     0      2      6     14     30     62    126    254    510   1022   2046   4094   8190  16382  32766  65534 131070]



sizes = [10, 20, 30, 40, 50, 60]

functions = OrderedDict()
functions['seq_prefixsum'] = seq_prefixsum
functions['multiproc_prefixsum'] = multiproc_prefixsum

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

scores = pd.DataFrame(data=0, columns=functions.keys(), index=sizes)
for size in sizes:
    for name, function in functions.items():
        data = np.asarray([2**i for i in range(1, size + 1)]).astype(np.uint32)
        wrapped = wrapper(function, data, size)
        result = timeit.timeit(wrapped, number=10)
        scores.loc[size, name] = result

print scores

