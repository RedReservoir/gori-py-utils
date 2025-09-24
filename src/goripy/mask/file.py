import numpy

import goripy.mask.rle



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
        mask_rle=goripy.mask.rle.mask_to_rle(mask),
        mask_shape=numpy.asarray([mask.shape[0], mask.shape[1]])
    )


def load_mask_from_rle_file(rle_filename):
    """
    Loads a binary mask from an RLE file.

    :param rle_filename: str
        Filename to load the mask from.
    """

    mask_rle_file = numpy.load(rle_filename)
    return goripy.mask.rle.rle_to_mask(mask_rle_file["mask_rle"], mask_rle_file["mask_shape"])
