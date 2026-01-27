import numpy



def crop_array_pads(
    array,
    pad_width
):
    """
    """

    num_dims = len(array.shape)
    num_pads = len(pad_width)

    if num_dims % num_pads != 0:
        raise ValueError("Array dimensions ({:d}) not divisible by pad dimensions ({:d}).".format(
            num_dims,
            num_pads
        ))

    pads_list = pad_width * (num_dims // num_pads)

    dim_slices = tuple(
        slice(pads[0], dim_size - pads[1])
        for dim_size, pads in zip(array.shape, pads_list)
    )

    return array[dim_slices]
