"""
Method argument processing utils.
"""



import numpy



def arg_list_to_arg_arr(
    arg,
    target_len,
    target_dtype
):
    """
    Validates and pre-processes a indefinite length sequence of numerical arguments.
    
    Args:

        arg (any):
            Argument(s) to validate and convert. Can be a single value or a sequence.

        target_len (int):
            Target number of numerical arguments.
            If possible, the original arguments sequence will be expanded to the target length by
            repeating the arguments as needed.
        
        target_dtype (numpy.dtype):
            Target argument data type.
            The elements of the original arguments sequence will be cast to this data type.
        
    Returns:

        numpy.ndarray:
            Validated and pre-processed arguments numpy array.
    """

    if isinstance(arg, (list, tuple, numpy.ndarray)):

        if target_len % len(arg) != 0:
            raise ValueError("Target length {:d} not dividible by arg size {:d}".format(
                target_len,
                len(arg)
            ))
        
        num_repeats = target_len // len(arg)
        arg_arr = numpy.asarray(arg * num_repeats, dtype=target_dtype)
    
    else:

        arg_arr = numpy.asarray([arg] * target_len, dtype=target_dtype)

    return arg_arr
