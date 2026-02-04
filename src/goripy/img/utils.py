import numpy
import skimage

import goripy.args



def compute_target_img_size_max(
    curr_size,
    max_size
):
    """
    Computes target image resizing size. Preserves aspect ratio and adjusts the target image size
    to be just under the maximum size.

    Args:

        curr_size (2-tuple of int):
            Current image size (H x W).

        max_size (2-tuple of int):
            Maximum image size (H x W).

    Returns:
    
        2-tuple of int:
            Target image size (H x W).
    """

    curr_h, curr_w = curr_size
    max_h, max_w = max_size

    ratio_h = max_h / curr_h
    ratio_w = max_w / curr_w

    if ratio_h < ratio_w:

        tgt_h = max_h
        tgt_w = round(curr_w * ratio_h)

    else:

        tgt_h = round(curr_h * ratio_w)
        tgt_w = max_w

    return tgt_h, tgt_w



def compute_target_img_size_min(
    curr_size,
    min_size
):
    """
    Computes target image resizing size. Preserves aspect ratio and adjusts the target image size
    to be just over the minimum size.

    Args:

        curr_size (2-tuple of int):
            Current image size (H x W).

        min_size (2-tuple of int):
            Maximum image size (H x W).

    Returns:
    
        2-tuple of int:
            Target image size (H x W).
    """

    curr_h, curr_w = curr_size
    min_h, min_w = min_size

    ratio_h = min_h / curr_h
    ratio_w = min_w / curr_w

    if ratio_h > ratio_w:

        tgt_h = min_h
        tgt_w = round(curr_w * ratio_h)

    else:

        tgt_h = round(curr_h * ratio_w)
        tgt_w = min_w

    return tgt_h, tgt_w



def compute_img_pad_sizes(
    curr_size,
    target_size
):
    """
    Computes required padding to pad an image to a target size.

    Args:

        curr_size (2-tuple of int):
            Current image size (H x W).

        target_size (2-tuple of int):
            Target image size (H x W).

    Returns:
    
        4-tuple of int:
            Required padding (top, bottom, left, right).
    """

    curr_h, curr_w = curr_size
    tgt_h, tgt_w = target_size

    if curr_h > tgt_h :
        raise ValueError("Current height exceeds target height")

    if curr_w > tgt_w :
        raise ValueError("Current width exceeds target width")

    pad_t = (tgt_h - curr_h) // 2
    pad_b = (tgt_h - curr_h) - pad_t

    pad_l = (tgt_w - curr_w) // 2
    pad_r = (tgt_w - curr_w) - pad_l

    return pad_t, pad_b, pad_l, pad_r



########



def img_to_rgb(
    img
):
    """
    Converts an image to RGB format.
    
    Requires input data type `numpy.uint8`.
    Output data type will also be `numpy.uint8`.

    Can handle the following shapes:
      - (H x W): Grayscale with no channel dimension.
      - (H x W x 1): Grayscale with channel dimension.
      - (H x W x 3): RGB.
      - (H x W x 4): RGBA.

    Args:

        img (numpy.ndarray):
            Original image.

    Returns:
    
        numpy.ndarray:
            New image in RGB format with (H x W x 3) shape.
    """

    if len(img.shape) == 3:

        if img.shape[2] == 1:
        
            img = skimage.color.gray2rgb(img[:, :, 0])
            return img

        elif img.shape[2] == 3:
        
            return img

        elif img.shape[2] == 4:
        
            img = skimage.color.rgba2rgb(img)
            img = skimage.util.img_as_ubyte(img)
            return img
        
    elif len(img.shape) == 2:

        img = skimage.color.gray2rgb(img)
        return img

    raise ValueError("Unexpected image shape: {:s}".format(str(img.shape)))



########



def resize_pad_fill_img(
    img,
    target_size,
    fill_value=0
):
    """
    Resizes and then pads an image to match a target size, preserving the aspect ratio.
    If the original data type of the image is `numpy.uint8`, it will be preserved.

    Args:

        img (numpy.ndarray):
            Original image.
            Must have shape (H x W) or (H x W x C).

        target_size (2-tuple of int or int):
            The target height and width of the image.
            If one value is provided, both height and width will be set to that value.

        fill_value (any):
            Value to fill the padding with. Must be compatible with the image channel-wise.
            Default: 0.

    Returns:
    
        numpy.ndarray:
            New image after resizing and padding.
    """

    # Pre-process arguments

    target_size = goripy.args.arg_list_to_arg_arr(
        target_size,
        target_len=2,
        target_dtype=numpy.uint16
    )
    target_size = tuple(map(int, target_size))

    fill_value = goripy.args.arg_list_to_arg_arr(
        fill_value,
        target_len=1 if len(img.shape) == 2 else img.shape[2],
        target_dtype=img.dtype
    )

    # Resize image preserving aspect ratio, if necessary

    img_size = (img.shape[0], img.shape[1])
    resize_size = compute_target_img_size_max(img_size, target_size)

    if resize_size != img_size:

        cast_to_ubyte = (img.dtype == numpy.uint8) 
        img = skimage.transform.resize(img, resize_size)
        if cast_to_ubyte: img = skimage.util.img_as_ubyte(img)

    # Pad image to target size and fill padding, if necessary
    
    pad_t, pad_b, pad_l, pad_r = compute_img_pad_sizes(resize_size, target_size)

    if (pad_t, pad_b, pad_l, pad_r) != (0, 0, 0, 0):

        if len(img.shape) == 2: pad_width = ((pad_t, pad_b), (pad_l, pad_r))
        if len(img.shape) == 3: pad_width = ((pad_t, pad_b), (pad_l, pad_r), (0, 0))

        img = numpy.pad(img, pad_width=pad_width)

        if pad_t > 0: img[:pad_t, :, :] = fill_value
        if pad_b > 0: img[-pad_b:, :, :] = fill_value
        if pad_l > 0: img[:, :pad_l, :] = fill_value
        if pad_r > 0: img[:, -pad_r:, :] = fill_value

    return img
