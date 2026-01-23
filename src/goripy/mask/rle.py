import numpy



def mask_to_rle(mask):
    """
    Encodes a binary mask into RLE.

    Args:

        mask (numpy.ndarray):
            The mask to encode.
            Shape: (H x W). Dtype: bool.

    Returns:

        list of int:
            The encoded RLE as an array.
            Dtype: uint32.
    """

    mask_flat = mask.ravel(order="F")

    rle = numpy.flatnonzero(
        numpy.logical_xor(mask_flat[1:], mask_flat[:-1])
    ).astype(numpy.uint32)
    
    rle = numpy.concatenate([
        rle + 1,
        numpy.asarray([mask_flat.shape[0]], dtype=numpy.uint32)
    ])
    
    rle[1:] -= rle[:-1]

    if mask_flat[0].item() == 1:
        rle = numpy.concatenate([
            numpy.asarray([0], dtype=numpy.uint32),
            rle
        ])
    
    return rle



def rle_to_mask(rle, shape):
    """
    Decodes a binary mask from RLE.

    Args:
    
        rle (numpy.ndarray):
            The encoded RLE as an array.
            Dtype: uint32.

        shape (2-tuple of int):
            The original dimensions of the mask (H x W).

    Returns:

        numpy.ndarray:
            The decoded mask.
            Shape: (H x W). Dtype: bool.
    """

    mask_flat = numpy.zeros(shape[0] * shape[1], dtype=bool)
    
    rle_cum = numpy.cumsum(rle)
    for idx_1, idx_2 in zip(rle_cum[::2], rle_cum[1::2]):
        mask_flat[idx_1:idx_2] = 1
    
    return mask_flat.reshape(shape, order='F')
