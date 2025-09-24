import numpy



def mask_to_rle(mask):
    """
    Encodes a binary mask into RLE.

    :param mask: numpy.ndarray
        2D boolean numpy array (H x W).

    :return: list of int
        RLE of the binary mask.
    """

    mask_flat = mask.ravel(order="F")

    rle = numpy.argwhere(numpy.logical_xor(mask_flat[1:], mask_flat[:-1])).flatten()
    rle = numpy.concatenate([rle + 1, [mask_flat.shape[0]]])
    rle[1:] -= rle[:-1]
    if mask_flat[0].item() == 1: rle = numpy.concatenate([[0], rle])
    
    return rle



def rle_to_mask(rle, shape):
    """
    Decodes a binary mask into RLE.

    :param rle: list of int
        RLE of the binary mask.
    :param shape: 2-tuple of int
        The original dimensions of the mask (H x W).

    :return: numpy.ndarray
        2D boolean numpy array (H x W).
    """

    mask_flat = numpy.zeros(shape[0] * shape[1], dtype=bool)
    
    rle_cum = numpy.cumsum(rle)
    for idx_1, idx_2 in zip(rle_cum[::2], rle_cum[1::2]):
        mask_flat[idx_1:idx_2] = 1
    
    return mask_flat.reshape(shape, order='F')



def save_mask_to_rle_file(mask, rle_filename):
    """
    Saves a binary mask into an RLE file.

    :param mask: numpy.ndarray
        2D boolean numpy array (H x W).
    :param rle_filename: str
        Filename to save the mask to.
    """

    numpy.savez(
        rle_filename,
        mask_rle=mask_to_rle(mask),
        mask_shape=numpy.asarray([mask.shape[0], mask.shape[1]])
    )


def load_mask_from_rle_file(rle_filename):
    """
    Loads a binary mask from an RLE file.

    :param rle_filename: str
        Filename to load the mask from.
    """

    mask_rle_file = numpy.load(rle_filename)
    return rle_to_mask(mask_rle_file["mask_rle"], mask_rle_file["mask_shape"])
