import numpy
import skimage

import albumentations
import cv2

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

        curr_size (2-tuple of int):
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

        curr_size (2-tuple of int):
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



def pad_fill_img(
    img,
    size,
    fill_value=0
):
    """
    Pads an image around the borders to a desired size.
    Can handle images of shape (H x W) or (H x W x C).
    
    Args:

        img (numpy.ndarray):
            Original image.

        size (2-tuple of int or int):
            The desired height and width of the padded image.
            If one value is provided, both height and width will be set to that value.

        fill_value (any, default=0):
            Value(s) to fill the padding with. Must be compatible with the image data type.

    Returns:
    
        numpy.ndarray:
            New image with padding.
    """

    # Check image shape

    if (len(img.shape) != 2) and (len(img.shape) != 3):
        raise ValueError("Unexpected image shape: {:s}".format(str(img.shape)))

    # Prepare args

    size = goripy.args.arg_list_to_arg_arr(size, 2).astype(numpy.uint16).tolist()

    fill_value = goripy.args.arg_list_to_arg_arr(
        fill_value, 1 if len(img.shape) == 2 else img.shape[2]
    ).astype(img.dtype)

    # Create padded image with fill value

    pad_img_shape = size if len(img.shape) == 2 else (size[0], size[1], img.shape[2])
    pad_img = numpy.empty(shape=pad_img_shape, dtype=img.dtype)

    # Fill padded image

    img_h, img_w = img.shape[0], img.shape[1]
    pad_img_h, pad_img_w = size

    copy_y0 = (pad_img_h - img_h) // 2
    copy_y1 = copy_y0 + img_h

    copy_x0 = (pad_img_w - img_w) // 2
    copy_x1 = copy_x0 + img_w

    ## Copy original image

    pad_img[copy_y0:copy_y1, copy_x0:copy_x1] = img[:, :]

    ## Fill border cornets

    pad_img[0:copy_y0, 0:copy_x0] = fill_value
    pad_img[0:copy_y0, copy_x1:pad_img_w] = fill_value
    pad_img[copy_y1:pad_img_h, 0:copy_x0] = fill_value
    pad_img[copy_y1:pad_img_h, copy_x1:pad_img_w] = fill_value

    ## Fill border rectangles

    pad_img[0:copy_y0, copy_x0:copy_x1] = fill_value
    pad_img[copy_y1:pad_img_h, copy_x0:copy_x1] = fill_value
    pad_img[copy_y0:copy_y1, 0:copy_x0] = fill_value
    pad_img[copy_y0:copy_y1, copy_x1:pad_img_w] = fill_value

    return pad_img



def img_to_rgb(
    img
):
    """
    Converts an image to RGB.
    
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
