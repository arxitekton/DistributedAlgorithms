import os, sys, time
import numpy as np
import pandas as pd
import math, random
import multiprocessing
from multiprocessing import Process, Manager
from collections import OrderedDict
import timeit


def msort(a):
    length_a = len(a)
    if length_a <= 1: return a
    m = int(math.floor(length_a / 2))
    a_left = a[0:m]
    a_right = a[m:]
    a_left = msort(a_left)
    a_right = msort(a_right)
    return merge(a_left, a_right)


def merge(left, right):
    a = []
    while len(left) > 0 or len(right) > 0:
        if len(left) > 0 and len(right) > 0:
            if left[0] <= right[0]:
                a.append(left.pop(0))
            else:
                a.append(right.pop(0))
        elif len(left) > 0:
            a.extend(left)
            break
        elif len(right) > 0:
            a.extend(right)
            break
    return a

# wrapper for one signature ## todo smth
def single_msort(a, cores):
    return msort(a)

def multiproc_msort(a, cores):

    # multiprocess Manager
    manager = Manager()
    responses = manager.list()

    # wrappers for mergesort
    def msort_multi(list_part):
        responses.append(msort(list_part))


    def merge_multi(list_part_left, list_part_right):
        responses.append(merge(list_part_left, list_part_right))

    #multiproc
    if cores > 1:

        step = int(math.floor(l / cores))
        p = []
        for n in range(0, cores):

            if n < cores - 1:
                proc = Process(target=msort_multi, args=(a[n * step:(n + 1) * step],))
            else:
                # get the remaining elements in the list
                proc = Process(target=msort_multi, args=(a[n * step:],))

            p.append(proc)

        for proc in p:
            proc.start()

        for proc in p:
            proc.join()

        p = []

        if len(responses) > 2:
            while len(responses) > 0:
                proc = Process(target=merge_multi, args=(responses.pop(0), responses.pop(0)))
                p.append(proc)
            for proc in p:
                proc.start()
            for proc in p:
                proc.join()

        a = merge(responses[0], responses[1])

        return a
    else:

        return msort(a)


if __name__ == '__main__':

    # length of our list
    l = 32
    print 'List length : ', l

    # create an unsorted list with random numbers
    a = [random.randint(0, n * 100) for n in range(0, l)]

    print 'generated random list: ', a

    # define number of cores (make it even)
    try:
        cores = multiprocessing.cpu_count()
        if cores > 1:
            if cores % 2 != 0:
                cores -= 1
        print 'Using %d cores' % cores
    except:
        cores = 1

    multip_result = multiproc_msort(a, cores)
    print "multip result:         ", multip_result

    single = msort(a)
    print "single result:         ", single

    print 'Arrays are equal:      ', (multip_result == single)

    # Compare

    sizes = [ 2**i for i in range(4, 17)]

    functions = OrderedDict()
    functions['single_msort'] = single_msort
    functions['multip_msort'] = multiproc_msort



    scores = pd.DataFrame(data=0, columns=functions.keys(), index=sizes)

    for size in sizes:

        data = [random.randint(0, n * 100) for n in range(0, size)]

        for name, function in functions.items():

            start_time = time.time()

            function(data, cores)

            finish = time.time() - start_time

            scores.loc[size, name] = finish

    print scores
