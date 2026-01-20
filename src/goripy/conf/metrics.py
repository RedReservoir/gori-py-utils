import numpy



def prec_metric_fun(tp, fp, fn, tn):
    """
    Computes the Precision metric using confusion matrix aggregates.

    Array dimensions:

    - C: Number of classes

    Args:

        tp (numpy.ndarray):
            A (C) dimensional array with the True Positive count per class.
        fp (numpy.ndarray):
            A (C) dimensional array with the False Positive count per class.
        fn (numpy.ndarray):
            A (C) dimensional array with the False Negative count per class.
        tn (numpy.ndarray):
            A (C) dimensional array with the True Negative count per class.

    Returns:

        numpy.ndarray:
            A (C) dimensional array with the Precision metric per class.
    """

    num = tp.copy()
    den = tp + fp

    metric = numpy.divide(num, den, where=den != 0)

    p = tp + fn
    n = fp + tn

    metric[(p == 0) | (n == 0)] = numpy.nan

    return metric



def rec_metric_fun(tp, fp, fn, tn):
    """
    Computes the Recall metric using confusion matrix aggregates.

    Array dimensions:

    - C: Number of classes

    Args:

        tp (numpy.ndarray):
            A (C) dimensional array with the True Positive count per class.
        fp (numpy.ndarray):
            A (C) dimensional array with the False Positive count per class.
        fn (numpy.ndarray):
            A (C) dimensional array with the False Negative count per class.
        tn (numpy.ndarray):
            A (C) dimensional array with the True Negative count per class.

    Returns:

        numpy.ndarray:
            A (C) dimensional array with the Recall metric per class.
    """

    num = tp.copy()
    den = tp + fn

    metric = numpy.divide(num, den, where=den != 0)

    p = tp + fn
    n = fp + tn

    metric[(p == 0) | (n == 0)] = numpy.nan

    return metric



def acc_metric_fun(tp, fp, fn, tn):
    """
    Computes the Accuracy metric using confusion matrix aggregates.

    Array dimensions:

    - C: Number of classes

    Args:

        tp (numpy.ndarray):
            A (C) dimensional array with the True Positive count per class.
        fp (numpy.ndarray):
            A (C) dimensional array with the False Positive count per class.
        fn (numpy.ndarray):
            A (C) dimensional array with the False Negative count per class.
        tn (numpy.ndarray):
            A (C) dimensional array with the True Negative count per class.

    Returns:

        numpy.ndarray:
            A (C) dimensional array with the Accuracy metric per class.
    """

    num = tp + tn
    den = tp + fp + fn + tn

    metric = numpy.divide(num, den, where=den != 0)

    metric[den == 0] = numpy.nan

    return metric



def f1_metric_fun(tp, fp, fn, tn):
    """
    Computes the F1-Score metric using confusion matrix aggregates.

    Array dimensions:

    - C: Number of classes

    Args:

        tp (numpy.ndarray):
            A (C) dimensional array with the True Positive count per class.
        fp (numpy.ndarray):
            A (C) dimensional array with the False Positive count per class.
        fn (numpy.ndarray):
            A (C) dimensional array with the False Negative count per class.
        tn (numpy.ndarray):
            A (C) dimensional array with the True Negative count per class.

    Returns:

        numpy.ndarray:
            A (C) dimensional array with the F1-Score metric per class.
    """

    num = 2 * tp
    den = (2 * tp) + fp + fn

    metric = numpy.divide(num, den, where=den != 0)

    p = tp + fn
    n = fp + tn

    metric[(p == 0) | (n == 0)] = numpy.nan

    return metric



def f1b_metric_fun(tp, fp, fn, tn, b=1.0):
    """
    Computes the FBeta-Score metric using confusion matrix aggregates.

    Array dimensions:

    - C: Number of classes

    Args:

        tp (numpy.ndarray):
            A (C) dimensional array with the True Positive count per class.
        fp (numpy.ndarray):
            A (C) dimensional array with the False Positive count per class.
        fn (numpy.ndarray):
            A (C) dimensional array with the False Negative count per class.
        tn (numpy.ndarray):
            A (C) dimensional array with the True Negative count per class.
        b (float):
            Value to use as "beta" in the FBeta-Score calculation.
            Meaning: Recall is "beta" times as important as Precision.
            Defaults to 1.0 (equal importance).

    Returns:

        numpy.ndarray:
            A (C) dimensional array with the FBeta-Score metric per class.
    """

    b2 = b * b

    num = (1 + b2) * tp
    den = ((1 + b2) * tp) + fp + (b2 * fn)

    metric = numpy.divide(num, den, where=den != 0)

    p = tp + fn
    n = fp + tn

    metric[(p == 0) | (n == 0)] = numpy.nan

    return metric
