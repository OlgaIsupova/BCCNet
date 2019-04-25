#  Copyright (c) 2019. University of Oxford

import numpy as np


def shuffle_arrays(arrays):
    if not arrays:
        return arrays

    size = arrays[0].shape[0]
    permutation = np.arange(size)
    np.random.shuffle(permutation)
    for i, array in enumerate(arrays):
        arrays[i] = array[permutation]

    return arrays


def shrink_arrays(arrays, shrink_size, is_shuffle=True):
    # if shrink_size in [0.0, 1.0] it specifies fraction of the array size to be extracted, if shrink_size is an
    # integer it specifies the size of the shrunk arrays

    if not arrays:
        return arrays

    if type(shrink_size) == float or type(shrink_size) == np.float64:
        assert(0.0 <= shrink_size <= 1.0)
        size = arrays[0].shape[0]
        shrunk_array_size = int(round(shrink_size * size))
    else:
        shrunk_array_size = shrink_size

    if is_shuffle:
        shuffled_arrays = shuffle_arrays(arrays)
    else:
        shuffled_arrays = arrays

    shrunk_arrays = []
    for array in shuffled_arrays:
        shrunk_arrays.append(array[:shrunk_array_size])

    return shrunk_arrays, shuffled_arrays


