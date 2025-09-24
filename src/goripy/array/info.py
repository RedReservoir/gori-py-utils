import numpy

import goripy.memory.info



def sprint_array_info(
    array,
    name=None
):
    """
    Generates array information into a string.
    Used for debug purposes.

    :param array: numpy.ndarray
        Tensor to print information from.
    :param name: str, optional
        Name of the array variable.
        If not provided, name will not be shown.
    
    :return: str
        A string with the array information.
    """

    array_info_str = ""

    if name is not None:
        array_info_str += "name: {:s}".format(name)
        array_info_str += ", "
    array_info_str += "shape: " + str(array.shape)
    array_info_str += ", "
    array_info_str += "dtype: " + str(array.dtype)
    array_info_str += ", "
    array_info_str += "mem: " + goripy.memory.info.sprint_fancy_num_bytes(array.nbytes)

    return array_info_str
