import numpy

import goripy.args



def unpad(
    array,
    pad_width
):
    """
    Crop an array. Inverse of `numpy.pad`.

    Args:

        array (numpy.ndarray):
            The array to unpad.
        
        pad_width (any):
            Number of values cropped to the edges of each axis. Functions similar to the
            `pad_width` argument in `numpy.pad`.

    Returns:

        numpy.ndarray:
            The unpadded array.
    """

    num_dims = len(array.shape)
    num_pads = len(pad_width)

    if num_dims % num_pads != 0:
        raise ValueError("Array dimensions ({:d}) not divisible by pad dimensions ({:d}).".format(
            num_dims,
            num_pads
        ))

    pads_list = [
        goripy.args.arg_list_to_arg_arr(pad_width_el, target_len=2, target_dtype=int)
        for pad_width_el in pad_width
    ] * (num_dims // num_pads)

    dim_slices = tuple(
        slice(pads[0], dim_size - pads[1])
        for dim_size, pads in zip(array.shape, pads_list)
    )

    return array[dim_slices]
