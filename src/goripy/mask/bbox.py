import numpy



def mask_to_bbox(mask):
    """
    Computes bbox limits from a binary mask.

    :param mask: numpy.ndarray
        2D boolean numpy array (H x W).

    :return: 4-tuple of int
        The bbox limits (x0, y0, x1, y1).
    """

    mask_x = numpy.any(mask, axis=0)
    x0, x1 = 0, mask_x.shape[0]
    while not mask_x[x0]: x0 += 1
    while not mask_x[x1 - 1]: x1 -= 1

    mask_y = numpy.any(mask, axis=1)
    y0, y1 = 0, mask_y.shape[0]
    while not mask_y[y0]: y0 += 1
    while not mask_y[y1 - 1]: y1 -= 1

    return x0, y0, x1, y1
