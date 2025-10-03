import os
import sys



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
    
    num_bytes = sys.getsizeof(obj)

    if type(obj) is list:
        for el in obj:
            num_bytes += get_obj_bytes(el)
    if type(obj) is dict:
        for key, val in obj.items():
            num_bytes += get_obj_bytes(key)
            num_bytes += get_obj_bytes(val)

    return num_bytes



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

    num_bytes = 0

    for subname in os.listdir(dirname):

        full_subname = os.path.join(dirname, subname)
        num_bytes += os.path.getsize(full_subname)

        if os.path.isdir(full_subname):
            num_bytes += get_dir_bytes(full_subname)

    return num_bytes
