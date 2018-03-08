import multiprocessing
import math

import numpy as np
import pandas as pd

import pycuda.autoinit
import pycuda.driver
import pycuda.compiler

import timeit

from collections import OrderedDict

n = 16
input_data = np.asarray([ i for i in range(1, n+1) ]).astype(np.uint32)


def seq_prefixsum(input_data, n):
    prefix_sum_cpu = np.zeros_like(input_data)

    for i in range(1,n):
        prefix_sum_cpu[i] = prefix_sum_cpu[i-1] + input_data[i-1]
    return prefix_sum_cpu


# gpu version
def gpu_prefixsum(input_data, n):
    # DEFINE block GPU
    source_module = pycuda.compiler.SourceModule \
            (
            """
            __global__ void prefix_sum_up_sweep( unsigned int* d_prefix_sum, int n, int d )
            {
                int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
                int k               = global_index_1d * ( 2 << d );

                int left_index;
                int right_index;

                if ( d == 0 )
                {
                    left_index  = k;
                    right_index = k + 1;
                }
                else
                {
                    left_index  = k + ( 2 << ( d - 1 ) ) - 1;
                    right_index = k + ( 2 << d )         - 1;
                }

                if ( right_index < n )
                {
                    d_prefix_sum[ right_index ] = d_prefix_sum[ left_index ] + d_prefix_sum[ right_index ];
                }
            }

            __global__ void prefix_sum_down_sweep( unsigned int* d_prefix_sum, int n, int d )
            {
                int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
                int k               = global_index_1d * ( 2 << d );

                int left_index;
                int right_index;

                if ( d == 0 )
                {
                    left_index  = k;
                    right_index = k + 1;
                }
                else
                {
                    left_index  = k + ( 2 << ( d - 1 ) ) - 1;
                    right_index = k + ( 2 << d )         - 1;
                }

                if ( right_index < n )
                {
                    unsigned int temp           = d_prefix_sum[ right_index ];
                    d_prefix_sum[ right_index ] = d_prefix_sum[ left_index ] + d_prefix_sum[ right_index ];
                    d_prefix_sum[ left_index ]  = temp;
                }
            }

            __global__ void blocked_prefix_sum_set_last_block_elements_to_zero( unsigned int* d_prefix_sums, int n, int block_size_num_elements )
            {
                int global_index_1d_left  = ( ( ( threadIdx.x * 2 ) + 1 ) * block_size_num_elements ) - 1;
                int global_index_1d_right = ( ( ( threadIdx.x * 2 ) + 2 ) * block_size_num_elements ) - 1;

                if ( global_index_1d_left < n )
                {
                    d_prefix_sums[ global_index_1d_left ] = 0;
                }

                if ( global_index_1d_right < n )
                {
                    d_prefix_sums[ global_index_1d_right ] = 0;
                }
            }

            __global__ void blocked_prefix_sum_down_sweep(
                unsigned int* d_prefix_sum,
                unsigned int* d_block_sums,
                unsigned int* d_input_data_resized,
                int n,
                int d )
            {
                int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;
                int k               = global_index_1d * ( 2 << d );

                int left_index;
                int right_index;

                if ( d == 0 )
                {
                    left_index  = k;
                    right_index = k + 1;
                }
                else
                {
                    left_index  = k + ( 2 << ( d - 1 ) ) - 1;
                    right_index = k + ( 2 << d )         - 1;
                }

                if ( right_index < n )
                {
                    unsigned int temp           = d_prefix_sum[ right_index ];
                    d_prefix_sum[ right_index ] = d_prefix_sum[ left_index ] + d_prefix_sum[ right_index ];
                    d_prefix_sum[ left_index ]  = temp;
                }

                if ( d == 0 && threadIdx.x == blockDim.x - 1 )
                {
                    d_block_sums[ blockIdx.x ] = d_prefix_sum[ right_index ] + d_input_data_resized[ right_index ];
                }
            }

            __global__ void blocked_prefix_sum_add_block_sums( unsigned int* d_prefix_sums, unsigned int* d_block_sums, int n )
            {
                int global_index_1d = 2 * ( ( blockIdx.x * blockDim.x ) + threadIdx.x );

                if ( blockIdx.x > 0 && global_index_1d < n - 1 )
                {
                    unsigned int block_sum               = d_block_sums[ blockIdx.x ];
                    d_prefix_sums[ global_index_1d ]     = d_prefix_sums[ global_index_1d ] + block_sum;
                    d_prefix_sums[ global_index_1d + 1 ] = d_prefix_sums[ global_index_1d + 1 ] + block_sum;
                }
            }
            """
        )

    block_size_num_elements = 1024
    block_size_num_threads = block_size_num_elements / 2

    num_elements_to_pad = 0
    if n % block_size_num_elements != 0:
        num_elements_to_pad = block_size_num_elements - (n % block_size_num_elements)

    input_data_resized_num_elements = n + num_elements_to_pad
    input_data_resized_num_threads = input_data_resized_num_elements / 2

    input_data_resized = np.zeros(input_data_resized_num_elements, dtype=input_data.dtype)
    input_data_resized[0:n] = input_data
    prefix_sum_gpu = np.zeros_like(input_data_resized)
    block_sums_gpu = np.zeros(block_size_num_elements, dtype=input_data_resized.dtype)
    input_data_resized_device = pycuda.driver.mem_alloc(input_data_resized.nbytes)
    prefix_sum_device = pycuda.driver.mem_alloc(prefix_sum_gpu.nbytes)
    block_sums_device = pycuda.driver.mem_alloc(block_sums_gpu.nbytes)

    prefix_sum_down_sweep_function = source_module.get_function("prefix_sum_down_sweep")
    prefix_sum_up_sweep_function = source_module.get_function("prefix_sum_up_sweep")
    blocked_prefix_sum_down_sweep_function = source_module.get_function("blocked_prefix_sum_down_sweep")
    blocked_prefix_sum_set_last_block_elements_to_zero_function = source_module.get_function(
        "blocked_prefix_sum_set_last_block_elements_to_zero")
    blocked_prefix_sum_add_block_sums_function = source_module.get_function("blocked_prefix_sum_add_block_sums")

    num_sweep_passes = int(math.ceil(math.log(block_size_num_elements, 2)))
    block_sums_gpu = np.zeros(block_size_num_elements, dtype=input_data_resized.dtype)

    pycuda.driver.memcpy_htod(input_data_resized_device, input_data_resized)
    pycuda.driver.memcpy_htod(prefix_sum_device, input_data_resized)
    pycuda.driver.memcpy_htod(block_sums_device, block_sums_gpu)

    #
    # block scan input array
    #
    prefix_sum_up_sweep_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(math.ceil(float(input_data_resized_num_threads) / float(prefix_sum_up_sweep_function_block[0])))
    prefix_sum_up_sweep_function_grid = (num_blocks, 1)

    blocked_prefix_sum_set_last_block_elements_to_zero_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(math.ceil(
        float(block_size_num_threads) / float(blocked_prefix_sum_set_last_block_elements_to_zero_function_block[0])))
    blocked_prefix_sum_set_last_block_elements_to_zero_function_grid = (num_blocks, 1)

    blocked_prefix_sum_down_sweep_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(
        math.ceil(float(input_data_resized_num_threads) / float(blocked_prefix_sum_down_sweep_function_block[0])))
    blocked_prefix_sum_down_sweep_function_grid = (num_blocks, 1)

    for d in range(num_sweep_passes):
        prefix_sum_up_sweep_function(
            prefix_sum_device,
            np.int32(input_data_resized_num_elements),
            np.int32(d),
            block=prefix_sum_up_sweep_function_block,
            grid=prefix_sum_up_sweep_function_grid)

    blocked_prefix_sum_set_last_block_elements_to_zero_function(
        prefix_sum_device,
        np.int32(input_data_resized_num_elements),
        np.int32(block_size_num_elements),
        block=blocked_prefix_sum_set_last_block_elements_to_zero_function_block,
        grid=blocked_prefix_sum_set_last_block_elements_to_zero_function_grid)

    for d in range(num_sweep_passes - 1, -1, -1):
        blocked_prefix_sum_down_sweep_function(
            prefix_sum_device,
            block_sums_device,
            input_data_resized_device,
            np.int32(input_data_resized_num_elements),
            np.int32(d),
            block=blocked_prefix_sum_down_sweep_function_block,
            grid=blocked_prefix_sum_down_sweep_function_grid)

    #
    # block scan block sums array
    #
    prefix_sum_up_sweep_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(math.ceil(float(block_size_num_threads) / float(prefix_sum_up_sweep_function_block[0])))
    prefix_sum_up_sweep_function_grid = (num_blocks, 1)

    blocked_prefix_sum_set_last_block_elements_to_zero_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(math.ceil(
        float(block_size_num_threads) / float(blocked_prefix_sum_set_last_block_elements_to_zero_function_block[0])))
    blocked_prefix_sum_set_last_block_elements_to_zero_function_grid = (num_blocks, 1)

    prefix_sum_down_sweep_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(math.ceil(float(block_size_num_threads) / float(prefix_sum_down_sweep_function_block[0])))
    prefix_sum_down_sweep_function_grid = (num_blocks, 1)

    for d in range(num_sweep_passes):
        prefix_sum_up_sweep_function(
            block_sums_device,
            np.int32(block_size_num_elements),
            np.int32(d),
            block=prefix_sum_up_sweep_function_block,
            grid=prefix_sum_up_sweep_function_grid)

    blocked_prefix_sum_set_last_block_elements_to_zero_function(
        block_sums_device,
        np.int32(block_size_num_elements),
        np.int32(block_size_num_elements),
        block=blocked_prefix_sum_set_last_block_elements_to_zero_function_block,
        grid=blocked_prefix_sum_set_last_block_elements_to_zero_function_grid)

    for d in range(num_sweep_passes - 1, -1, -1):
        prefix_sum_down_sweep_function(
            block_sums_device,
            np.int32(block_size_num_elements),
            np.int32(d),
            block=prefix_sum_down_sweep_function_block,
            grid=prefix_sum_down_sweep_function_grid)

    #
    # distribute scanned block sums back into the prefix sums
    #

    blocked_prefix_sum_add_block_sums_function_block = (block_size_num_threads, 1, 1)
    num_blocks = int(
        math.ceil(float(input_data_resized_num_threads) / float(blocked_prefix_sum_add_block_sums_function_block[0])))
    blocked_prefix_sum_add_block_sums_function_grid = (num_blocks, 1)

    blocked_prefix_sum_add_block_sums_function(
        prefix_sum_device,
        block_sums_device,
        np.int32(input_data_resized_num_elements),
        block=blocked_prefix_sum_add_block_sums_function_block,
        grid=blocked_prefix_sum_add_block_sums_function_grid)

    #
    # copy data back to host
    #

    pycuda.driver.memcpy_dtoh(prefix_sum_gpu, prefix_sum_device)

    return prefix_sum_gpu[0:n]


# some test
print "input_data:"
print input_data
print "seq_prefixsum:"
print seq_prefixsum(input_data, n)
print "gpu_prefixsum:"
print gpu_prefixsum(input_data, n)

# input_data:
# [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
# seq_prefixsum:
# [  0   1   3   6  10  15  21  28  36  45  55  66  78  91 105 120]
# gpu_prefixsum:
# [  0   1   3   6  10  15  21  28  36  45  55  66  78  91 105 120]



sizes = [1000, 10000, 10000, 100000, 1000000]

functions = OrderedDict()
functions['seq_prefixsum'] = seq_prefixsum
functions['gpu_prefixsum'] = gpu_prefixsum

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

scores = pd.DataFrame(data=0, columns=functions.keys(), index=sizes)
for size in sizes:
    for name, function in functions.items():
        data = np.asarray([i for i in range(1, size + 1)]).astype(np.uint32)
        wrapped = wrapper(function, data, size)
        result = timeit.timeit(wrapped, number=10)
        scores.loc[size, name] = result

print scores

#          seq_prefixsum  gpu_prefixsum
# 1000          0.002481       0.014228
# 10000         0.024503       0.016712
# 10000         0.024503       0.016712
# 100000        0.250679       0.020041
# 1000000       2.553071       0.068524
