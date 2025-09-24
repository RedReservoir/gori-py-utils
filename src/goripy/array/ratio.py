import numpy



def discrete_ratio_partition_sizes(total, ratios):
    """
    Computes discrete partition sizes.

    :param total: int
        Total number of elemens.
    :param ratios: numpy.ndarray
        The ratios for each partition.
        All ratios will automatically be normalized to sum up to 1.

    :return: numpy.ndarray
        The sizes for each partition.
    """

    norm_ratios = ratios / numpy.sum(ratios)

    full_val_arr = norm_ratios * total
    round_val_arr = numpy.around(norm_ratios * total).astype(int)

    diff_list_arr = round_val_arr - full_val_arr

    total_diff = total - numpy.sum(round_val_arr)

    argsort_diff_list_arr = numpy.argsort(diff_list_arr)

    if total_diff > 0: round_val_arr[argsort_diff_list_arr[:total_diff]] += 1
    if total_diff < 0: round_val_arr[argsort_diff_list_arr[total_diff:]] -= 1

    return round_val_arr
