import math

import numpy



def chunk_partition_size(arr, chunk_size):
    """
    Partitions a numpy array into sub-arrays.

    Args:

        arr (numpy.ndarray):
            1D numpy array to partition.

        chunk_size (int):
            Desired size of the sub-arrays.

    Returns:

        list of numpy.ndarray:
            List containing the sub-arrays.
    """

    return numpy.array_split(arr, max(1, math.ceil(arr.shape[0] / chunk_size)))



def chunk_partition_num(arr, num_chunks):
    """
    Partitions a numpy array into sub-arrays.

    Args:

        arr (numpy.ndarray):
            1D numpy array to partition.

        num_chunks (int):
            Desired number of chunks.

    Returns:

        list of numpy.ndarray:
            List containing the sub-arrays.
    """

    return numpy.array_split(arr, num_chunks)



def chunk_partition_size_list(my_list, chunk_size):
    """
    Partitions a list into sub-lists.

    Args:

        my_list (list):
            List to partition.

        chunk_size (int):
            Desired size of the sub-lists.

    Returns:

        list of list:
            List containing the sub-lists.
    """

    return [
        [my_list[idx] for idx in idx_chunk]
        for idx_chunk in chunk_partition_size(numpy.arange(len(my_list)), chunk_size)
    ]



def chunk_partition_num_list(my_list, num_chunks):
    """
    Partitions a list into sub-lists.

    Args:

        my_list (list):
            List to partition.

        num_chunks (int):
            Desired number of chunks.

    Returns:
    
        list of list:
            List containing the sub-lists.
    """

    return [
        [my_list[idx] for idx in idx_chunk]
        for idx_chunk in chunk_partition_num(numpy.arange(len(my_list)), num_chunks)
    ]
