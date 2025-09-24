import numpy



def first_argwhere_zidxs(arr, vals):
    """
    Computes the indices of the first occurrence of values in an original array.

    Args:

        arr (numpy.ndarray):
            1D original array to search in.

        vals (numpy.ndarray):
            1D array with the values to search.

    Returns:

        numpy.ndarray:
            The computed first occurrence indices.
            The following satisfies: `arr[zidxs] = vals`.
    """

    zidxs = []
    for val in vals:
        argw = numpy.argwhere(arr == val)
        if argw.shape[0] == 0:
            raise ValueError("Value {:s} not found in arr".format(str(val)))
        zidxs.append(argw[0][0])

    return numpy.asarray(zidxs)



def discrete_linspace_sizes(start, stop, num):
    """
    Computes sizes of a discretized linspace.

    Args:

        start (int):
            The starting value of the sequence.

        stop (int):
            The ending value of the sequence.

        num_steps (int):
            Number of samples to generate. Must be non-negative.

    Returns:
    
        numpy.ndarray:
            Discretized linspace sizes.
    """

    disc_linspace_limits = numpy.round(numpy.linspace(start, stop, num)).astype("int")
    disc_linspace_vals = disc_linspace_limits[1:] - disc_linspace_limits[:-1]

    return disc_linspace_vals
