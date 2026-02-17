import numpy



def llist_to_arrr(
    llist,
    arrr_dtype,
    arrr_dim1_size=None,
    arrr_inv_val=None
):
    """
    Creates a 2D numpy array and fills it with data coming from a 2D list.

    Args:

        llist (list of list of any):
            2D list to fill the array with.

        arrr_dtype (numpy.dtype):
            Resulting numpy array dtype.

        arrr_dim1_size (int, optional):
            Resulting array axis 1 size. If not provided, the size of the largest sublist will be
            used as axis 1 size.

        arrr_inv_val (any, optional):
            Valid to use as padding in the resulting array. If not provided, the maximum value of
            the numpy dtype will be used. Must be provided if the numpy dtype is neither integer
            or floating.    
        
    Returns:

        numpy.ndarray:
            Resulting numpy array filled with the 2D list data.
    """


    if arrr_dim1_size is None:
        arrr_dim1_size = max(len(llist_el) for llist_el in llist)

    #

    if arrr_inv_val is None:
        
        if numpy.issubdtype(arrr_dtype, numpy.integer):
            arrr_inv_val = numpy.iinfo(arrr_dtype).max

        elif numpy.issubdtype(arrr_dtype, numpy.floating):
            arrr_inv_val = numpy.finfo(arrr_dtype).max

        else:
            raise ValueError("Argument \"{:s}\" must be provided for dtype \"{:s}\"".format(
                "arr_inv_val",
                str(arrr_dtype)
            ))

    #

    arrr = numpy.full(
        shape=(len(llist), arrr_dim1_size),
        dtype=arrr_dtype,
        fill_value=arrr_inv_val
    )

    for idx, llist_el in enumerate(llist):
        arrr[idx, :len(llist_el)] = llist_el
    
    return arrr
