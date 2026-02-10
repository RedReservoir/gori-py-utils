import math

import numpy



def chunk_partition_size(arr, chunk_size):
    """
    Partitions a numpy array into sub-arrays, ensuring:
      - The size of all sub-arrays is as similar as possible.
      - The size of all sub-arrays is always under a desired specified size.

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
    Partitions a numpy array into a given number of sub-arrays, ensuring:
      - The size of all sub-arrays is as similar as possible.

    Args:

        arr (numpy.ndarray):
            1D numpy array to partition.

        num_chunks (int):
            Desired number of sub-arrays.

    Returns:

        list of numpy.ndarray:
            List containing the sub-arrays.
    """

    return numpy.array_split(arr, num_chunks)



def chunk_partition_size_list(my_list, chunk_size):
    """
    Partitions a list into sub-lists, ensuring:
      - The size of all sub-lists is as similar as possible.
      - The size of all sub-lists is always under a desired specified size.

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
    Partitions a list into a given number of sub-lists, ensuring:
      - The size of all sub-lists is as similar as possible.

    Args:

        my_list (list):
            List to partition.

        num_chunks (int):
            Desired number of sub-lists.

    Returns:
    
        list of list:
            List containing the sub-lists.
    """

    return [
        [my_list[idx] for idx in idx_chunk]
        for idx_chunk in chunk_partition_num(numpy.arange(len(my_list)), num_chunks)
    ]
