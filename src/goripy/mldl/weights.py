import numpy



def compute_freq_weights(
    freq_arr,
    freq_pow=1.0,
    freq_ratio_arr=None
):
    """
    Computes item weights based on inverse frequency.
    Takes into account item counts.

    Args:
    
        freq_arr (numpy.ndarray):
            1D array with the frequency values.

        freq_pow (float, optional):
            Power of reciprocal for frequency weight calculation.
            Defaults to 1.0.

        freq_ratio_arr (numpy.ndarray, optional):
            1D numpy array with the global item ratios. This array will be internally normalized.
            If not provided, uniform weights will be used.

    Returns:

        numpy.ndarray:
            The computed weight values.
    """

    # Compute default parameters

    if freq_ratio_arr is None:
        freq_ratio_arr = numpy.ones_like(freq_arr, dtype=float)
    else:
        freq_ratio_arr *= freq_ratio_arr.shape[0] / numpy.sum(freq_ratio_arr)

    # Compute weights

    with numpy.errstate(divide='ignore'):
        wt_arr = freq_arr.astype("float") ** (-freq_pow)

    wt_arr[numpy.isinf(wt_arr)] = 0

    wt_arr *= freq_ratio_arr

    corr_const = numpy.sum(freq_arr) / numpy.sum(freq_arr * wt_arr)
    wt_arr *= corr_const

    return wt_arr



def compute_pos_neg_freq_weights(
    pos_freq_arr,
    neg_freq_arr,
    freq_pow=1.0,
    pos_neg_pow=1.0,
    freq_ratio_arr=None,
    pos_neg_ratio=1.0
):
    """
    Computes item weights based on inverse frequency.
    Takes into account item counts and pos/neg counts.

    Args:

        pos_freq_arr (numpy.ndarray):
            1D array with the positive frequency values.

        neg_freq_arr (numpy.ndarray):
            1D array with the negative frequency values.

        freq_pow (float, optional):
            Power of reciprocal for frequency weight calculation.
            Defaults to 1.0.

        pos_neg_pow (float, optional):
            Power of reciprocal for positive/negative frequency weight calculation.
            Defaults to 1.0.

        freq_ratio_arr (numpy.ndarray, optional):
            1D numpy array with the global item ratios. This array will be internally normalized.
            If not provided, uniform weights will be used.

        pos_neg_ratio (float, optional):
            Ratio of positive global weight over negative global weight.
            Defaults to 1.0.

    Returns:

        tuple of numpy.ndarray:
            A tuple containing:
            
            - pos_wt_arr: The computed positive weight values.
            - wt_arr: The computed overall weight values.
    """

    # Compute default parameters

    if freq_ratio_arr is None:
        freq_ratio_arr = numpy.ones_like(pos_freq_arr, dtype=float)
    else:
        freq_ratio_arr *= freq_ratio_arr.shape[0] / numpy.sum(freq_ratio_arr)

    # Compute weights

    freq_arr = pos_freq_arr + neg_freq_arr

    with numpy.errstate(divide='ignore'):
        freq_wt_arr = freq_arr.astype("float") ** (-freq_pow)
        pos_wt_arr = pos_freq_arr.astype("float") ** (-pos_neg_pow)
        neg_wt_arr = neg_freq_arr.astype("float") ** (-pos_neg_pow)

    freq_wt_arr[numpy.isinf(freq_wt_arr)] = 0
    pos_wt_arr[numpy.isinf(pos_wt_arr)] = 0
    neg_wt_arr[numpy.isinf(neg_wt_arr)] = 0

    pos_ratio = 2 * pos_neg_ratio / (pos_neg_ratio + 1)
    neg_ratio = 2 / (pos_neg_ratio + 1)

    pos_wt_arr *= freq_wt_arr * pos_ratio * freq_ratio_arr
    neg_wt_arr *= freq_wt_arr * neg_ratio * freq_ratio_arr

    corr_const = numpy.sum(freq_arr) / numpy.sum((pos_freq_arr * pos_wt_arr) + (neg_freq_arr * neg_wt_arr))
    pos_wt_arr *= corr_const
    neg_wt_arr *= corr_const

    return pos_wt_arr, neg_wt_arr
