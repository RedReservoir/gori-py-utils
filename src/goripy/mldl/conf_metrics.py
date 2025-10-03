import numpy



def prec_metric_fun(tp, fp, fn, tn):
    """
    Computes the Precision metric using confusion matrix aggregates.

    :param tp: numpy.ndarray
        A (num_classes) array with the True Positive count per class.
    :param fp: numpy.ndarray
        A (num_classes) array with the False Positive count per class.
    :param fn: numpy.ndarray
        A (num_classes) array with the False Negative count per class.
    :param tn: numpy.ndarray
        A (num_classes) array with the True Negative count per class.

    :return: numpy.ndarray
        A (num_classes) array with the Precision metric per class.
    """

    num = tp
    den = tp + fp
    metric = numpy.divide(num, den, where=den != 0)
    metric[den == 0] = numpy.nan
    return metric


def rec_metric_fun(tp, fp, fn, tn):
    """
    Computes the Recall metric using confusion matrix aggregates.

    :param tp: numpy.ndarray
        A (num_classes) array with the True Positive count per class.
    :param fp: numpy.ndarray
        A (num_classes) array with the False Positive count per class.
    :param fn: numpy.ndarray
        A (num_classes) array with the False Negative count per class.
    :param tn: numpy.ndarray
        A (num_classes) array with the True Negative count per class.

    :return: numpy.ndarray
        A (num_classes) array with the Recall metric per class.
    """

    num = tp
    den = tp + fn
    metric = numpy.divide(num, den, where=den != 0)
    metric[den == 0] = numpy.nan
    return metric


def acc_metric_fun(tp, fp, fn, tn):
    """
    Computes the Accuracy metric using confusion matrix aggregates.

    :param tp: numpy.ndarray
        A (num_classes) array with the True Positive count per class.
    :param fp: numpy.ndarray
        A (num_classes) array with the False Positive count per class.
    :param fn: numpy.ndarray
        A (num_classes) array with the False Negative count per class.
    :param tn: numpy.ndarray
        A (num_classes) array with the True Negative count per class.

    :return: numpy.ndarray
        A (num_classes) array with the Accuracy metric per class.
    """

    num = tp + tn
    den = tp + fp + fn + tn
    metric = numpy.divide(num, den, where=den != 0)
    metric[den == 0] = numpy.nan
    return metric


def f1_metric_fun(tp, fp, fn, tn):
    """
    Computes the F1-Score metric using confusion matrix aggregates.

    :param tp: numpy.ndarray
        A (num_classes) array with the True Positive count per class.
    :param fp: numpy.ndarray
        A (num_classes) array with the False Positive count per class.
    :param fn: numpy.ndarray
        A (num_classes) array with the False Negative count per class.
    :param tn: numpy.ndarray
        A (num_classes) array with the True Negative count per class.

    :return: numpy.ndarray
        A (num_classes) array with the F1-Score metric per class.
    """

    num = 2 * tp
    den = (2 * tp) + fp + fn
    metric = numpy.divide(num, den, where=den != 0)
    metric[den == 0] = numpy.nan
    return metric


def f1b_metric_fun(tp, fp, fn, tn, b=1.0):
    """
    Computes the FBeta-Score metric using confusion matrix aggregates.

    :param tp: numpy.ndarray
        A (num_classes) array with the True Positive count per class.
    :param fp: numpy.ndarray
        A (num_classes) array with the False Positive count per class.
    :param fn: numpy.ndarray
        A (num_classes) array with the False Negative count per class.
    :param tn: numpy.ndarray
        A (num_classes) array with the True Negative count per class.
    :param b: float, default=1.0
        Value to use as "beta" in the FBeta-Score calculation.
        Meaning: Recall is "beta" times as important as Precision.

    :return: numpy.ndarray
        A (num_classes) array with the FBeta-Score metric per class.
    """

    b2 = b * b
    num = (1 + b2) * tp
    den = ((1 + b2) * tp) + fp + (b2 * fn)
    metric = numpy.divide(num, den, where=den != 0)
    metric[den == 0] = numpy.nan
    return metric



def compute_conf_metric_arr(
    conf_aggs,
    metric_fun,
    pn_weighted=True
):
    """
    Computes a confusion metric array using confusion matrix aggregates.

    :param conf_aggs: numpy.ndarray
        A (num_classes x 4) numpy array with the confusion matrix aggregates.
        Aggregates are expected in this order: tp, fp, fn, tn.
    :param metric_fun: callable
        A function that takes confusion aggregates and returns a metric.
    :param pn_weighted: bool, default=True
        If True, balances amount of positive and negative examples (target).
    
    :return: float or numpy.ndarray
        A (num_classes) numpy array with the confusion metric for each class.
    """

    tp_arr = conf_aggs[:, 0].copy()
    fp_arr = conf_aggs[:, 1].copy()
    fn_arr = conf_aggs[:, 2].copy()
    tn_arr = conf_aggs[:, 3].copy()

    if pn_weighted:

        p_arr = tp_arr + fn_arr
        n_arr = fp_arr + tn_arr

        tp_arr = numpy.divide(tp_arr.astype(float), p_arr, where=p_arr != 0)
        fp_arr = numpy.divide(fp_arr.astype(float), n_arr, where=n_arr != 0)
        fn_arr = numpy.divide(fn_arr.astype(float), p_arr, where=p_arr != 0)
        tn_arr = numpy.divide(tn_arr.astype(float), n_arr, where=n_arr != 0)
    
    metric_arr = metric_fun(tp_arr, fp_arr, fn_arr, tn_arr)

    return metric_arr



def compute_conf_metric_avg(
    conf_metric_arr,
    conf_aggs,
    average=None
):
    """
    :param conf_metric_arr: numpy.ndarray
        A (num_classes) numpy array with the metric per class.
    :param conf_aggs: numpy.ndarray
        A (num_classes x 4) numpy array with the confusion matrix aggregates.
        Aggregates are expected in this order: tp, fp, fn, tn.
    :param average: str
        Possible values:
          - "macro": Averages metric giving equal weight to each class.
          - "micro": Averages metric weighting each class by frequency.
    
    :return: float
        The averaged confusion metric.
    """

    if average is None:
        raise ValueError("Provide the `average` parameter")

    if average == "macro":
        
        metric = numpy.nanmean(conf_metric_arr)

    if average == "micro":
        
        tp_arr = conf_aggs[:, 0].copy()
        fn_arr = conf_aggs[:, 2].copy()

        p_arr = tp_arr + fn_arr
        metric = numpy.sum(conf_metric_arr * p_arr) / numpy.sum(p_arr)
        
    return metric
