import os
import sys

import numpy



def get_obj_bytes(obj):
    """
    Computes the number of bytes that an object weights.
    Supports basic data types, lists and dicts.

    Args:

        obj (any):
            Object to compute memory from.

    Returns:
    
        int:
            Number of bytes that the object weights.
    """

    if hasattr(obj, "get_num_bytes"):
        obj_num_bytes = obj.get_num_bytes()
    elif type(obj) is numpy.ndarray:
        obj_num_bytes = obj.nbytes
    elif type(obj) is list:
        obj_num_bytes = sys.getsizeof(obj)
        for el in obj:
            obj_num_bytes += get_obj_bytes(el)
    elif type(obj) is dict:
        obj_num_bytes = sys.getsizeof(obj)
        for key, val in obj.items():
            obj_num_bytes += get_obj_bytes(key)
            obj_num_bytes += get_obj_bytes(val)
    else:
        obj_num_bytes = sys.getsizeof(obj)

    return obj_num_bytes



def get_dir_bytes(dirname):
    """
    Computes the size of a directory and all its contents.

    Args:
        dirname (str):
            Directory to scan.

    Returns:
        int:
            Number of bytes that the directory weights.
    """

    dir_num_bytes = 0

    for subname in os.listdir(dirname):

        full_subname = os.path.join(dirname, subname)
        dir_num_bytes += os.path.getsize(full_subname)

        if os.path.isdir(full_subname):
            dir_num_bytes += get_dir_bytes(full_subname)

    return dir_num_bytes
