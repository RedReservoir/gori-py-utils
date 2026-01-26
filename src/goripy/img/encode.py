import io
import base64

import numpy

from PIL import Image



def encode_rgb_img(
    img,
    jpeg_quality=90
):
    """
    Encodes an RGB image into a Base64 string. Useful for HTTP data transfer.
    
    Internally converts the image into bytes via the JPEG algorithm, and then converts those bytes
    into a Base64 string.

    Args:

        img (numpy.ndarray):
            Image to encode.
            Shape: (H x W x 3). Dtype: `numpy.uint8`.

        jpeg_quality (int):
            JPEG compression quality value. Must lie in the [0, 100] range.
            Defaults to 90.
        
    Returns:

        str:
            The Base64 string encoding the image.
    """

    pil_img = Image.fromarray(img, mode="RGB")

    img_bytes_buf = io.BytesIO()
    pil_img.save(img_bytes_buf, format="JPEG", quality=jpeg_quality, optimize=True)

    b64_bytes = base64.b64encode(img_bytes_buf.getvalue())
    b64_str = b64_bytes.decode("ascii")

    return b64_str



def decode_rgb_img(
    b64_str
):
    """
    Decodes an RGB image from a Base64 string. Useful for HTTP data transfer.

    Decoding method associated to the `encode_rgb_img` encoding method.

    Args:

        b64_str (str):
            Base64 string encoding the image.
        
    Returns:

        numpy.ndarray:
            The decoded image.
            Shape: (H x W x 3). Dtype: `numpy.uint8`.
    """

    b64_bytes = base64.b64decode(b64_str)
    img_bytes_buf = io.BytesIO(b64_bytes)

    pil_img = Image.open(img_bytes_buf).convert("RGB")

    img = numpy.asarray(pil_img, dtype=numpy.uint8)

    return img
