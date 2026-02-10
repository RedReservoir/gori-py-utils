import numpy

import goripy.memory.info



def sprint_array_info(
    array,
    name=None
):
    """
    Prints array information to a string.

    Args:
    
        array (numpy.ndarray):
            Array to print information from.

        name (str, optional):
            Name of the array variable.
            If not provided, name will not be shown.

    Returns:

        str:
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
