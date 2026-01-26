import base64

import numpy

from goripy.mask.rle import mask_to_rle, rle_to_mask



def encode_mask(
    mask
):
    """
    Encodes a binary mask into a Base64 string. Useful for HTTP data transfer.
    
    Internally encodes the mask RLE encoding into a Base64 string.

    Args:

        mask (numpy.ndarray):
            Binary mask to encode.
            Shape: (H x W). Dtype: `bool`.
        
    Returns:

        str:
            The Base64 string encoding the binary mask.
    """

    rle_arr = mask_to_rle(mask)
    rle_bytes = rle_arr.tobytes()
    
    shape_str = "{:04d}{:04d}".format(mask.shape[0], mask.shape[1])
    shape_bytes = shape_str.encode()
    
    b64_bytes = base64.b64encode(rle_bytes + shape_bytes)
    b64_str = b64_bytes.decode("ascii")

    return b64_str



def decode_mask(
    b64_str
):
    """
    Decodes a binary mask from a Base64 string. Useful for HTTP data transfer.

    Decoding method associated to the `decode_mask` encoding method.

    Args:

        b64_str (str):
            Base64 string encoding the binary mask.
        
    Returns:

        numpy.ndarray:
            The decoded binary mask.
            Shape: (H x W). Dtype: `bool`.
    """
    
    b64_bytes = base64.b64decode(b64_str)

    rle_arr = numpy.frombuffer(b64_bytes[:-8], dtype=numpy.uint32)

    shape_str = b64_bytes[-8:].decode()
    shape = (int(shape_str[-8:-4]), int(shape_str[-4:]))

    mask = rle_to_mask(rle_arr, shape)

    return mask
