import numpy



def arg_list_to_arg_arr(
    arg,
    target_len
):

    if isinstance(arg, (list, tuple)):

        if target_len % len(arg) != 0:
            raise ValueError("Target length {:d} not dividible by arg size {:d}".format(
                target_len,
                len(arg)
            ))
        
        num_repeats = target_len // len(arg)
        arg_arr = numpy.asarray(arg * num_repeats)
    
    else:

        arg_arr = numpy.asarray([arg] * target_len)

    return arg_arr
