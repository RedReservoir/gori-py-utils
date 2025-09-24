import numpy



def mask_to_rle(mask):
    """
    Encodes a binary mask into RLE.

    Args:

        mask (numpy.ndarray):
            2D boolean numpy array (H x W).

    Returns:

        list of int:
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
    Decodes a binary mask from RLE.

    Args:
    
        rle (list of int):
            RLE of the binary mask.

        shape (tuple of int):
            The original dimensions of the mask (H x W).

    Returns:

        numpy.ndarray:
            2D boolean numpy array (H x W).
    """

    mask_flat = numpy.zeros(shape[0] * shape[1], dtype=bool)
    
    rle_cum = numpy.cumsum(rle)
    for idx_1, idx_2 in zip(rle_cum[::2], rle_cum[1::2]):
        mask_flat[idx_1:idx_2] = 1
    
    return mask_flat.reshape(shape, order='F')
