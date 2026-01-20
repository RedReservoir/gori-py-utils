import numpy



def pn_normalize_conf_aggs(
    conf_aggs
):
    """
    Normalizes confusion matrix aggregates wrt. the amount of positive and negative examples.

    Array dimensions:

    - C: Number of classes

    Args:

        conf_aggs (numpy.ndarray):
            A (C x 4) dimensional array with the original confusion matrix aggregates.
            Aggregates are expected in this order: tp, fp, fn, tn.
    
    Returns

        numpy.ndarray:
            A (C x 4) dimensional array with the normalized confusion matrix aggregates.
            Aggregates are expected in this order: tp, fp, fn, tn.
    """

    conf_aggs = conf_aggs.astype(float)

    p_arr = conf_aggs[:, 0] + conf_aggs[:, 2]
    n_arr = conf_aggs[:, 1] + conf_aggs[:, 3]

    conf_aggs[:, 0] = numpy.divide(conf_aggs[:, 0], p_arr, where=p_arr != 0)
    conf_aggs[:, 1] = numpy.divide(conf_aggs[:, 1], n_arr, where=n_arr != 0)
    conf_aggs[:, 2] = numpy.divide(conf_aggs[:, 2], p_arr, where=p_arr != 0)
    conf_aggs[:, 3] = numpy.divide(conf_aggs[:, 3], n_arr, where=n_arr != 0)

    conf_aggs[:, 0][p_arr == 0] = 0
    conf_aggs[:, 1][n_arr == 0] = 0
    conf_aggs[:, 2][p_arr == 0] = 0
    conf_aggs[:, 3][n_arr == 0] = 0

    return conf_aggs



def compute_conf_metric_arr(
    conf_aggs,
    metric_fun
):
    """
    Computes a confusion metric array using confusion matrix aggregates.

    Array dimensions:

    - C: Number of classes

    Args:

        conf_aggs (numpy.ndarray):
            A (C x 4) dimensional array with the confusion matrix aggregates.
            Aggregates are expected in this order: tp, fp, fn, tn.

        metric_fun (callable):
            A function that takes confusion aggregates and returns a metric.

        pn_weighted (bool):
            If True, balances amount of positive and negative examples (target).
            Defaults to True.
    
    Returns

        numpy.ndarray:
            A (C) dimensional array with the confusion metric for each class.
    """

    metric_arr = metric_fun(
        conf_aggs[:, 0],
        conf_aggs[:, 1],
        conf_aggs[:, 2],
        conf_aggs[:, 3]
    )

    return metric_arr



def compute_conf_metric_avg(
    conf_metric_arr,
    conf_aggs,
    avg_method
):
    """
    Averages confusion metrics computed per-class.

    Array dimensions:

    - C: Number of classes

    Args:

        conf_metric_arr (numpy.ndarray):
            A (C) dimensional array with the confusion metric for each class.

        conf_aggs (numpy.ndarray):
            A (C x 4) dimensional array with the confusion matrix aggregates.
            Aggregates are expected in this order: tp, fp, fn, tn.

        avg_method (str):
            Possible values:

            - "macro": Averages metric giving equal weight to each class.
            - "micro": Averages metric weighting each class by frequency.
        
    Returns

        float:
            The averaged confusion metric.
    """

    if avg_method == "macro":
        
        metric = numpy.nanmean(conf_metric_arr)

    elif avg_method == "micro":

        p_arr = conf_aggs[:, 0] + conf_aggs[:, 2]
        metric = numpy.nansum(conf_metric_arr * p_arr) / numpy.sum(p_arr)

    else:

        raise ValueError("Unrecognized avg_method \"{:s}\"".format(
            avg_method
        ))

    return metric
