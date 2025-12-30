import numpy
import skimage

import albumentations
import cv2

import goripy.args



def pad_resize(
    arr,
    size,
    fill_value=255
):
    """
    Pads and then resizes an array, preserving the aspect ratio.

    Args:

        arr (numpy.ndarray):
            Numpy array to pad and resize.
            Must have shape (H x W) or (H x W x C).

        size (2-tuple of int or int):
            The target height and width of the image.
            If one value is provided, both height and width will be set to that value.

        fill_value (any):
            Value to fill the array with. Must be compatible with the array.

    Returns:
    
        numpy.ndarray:
            New array after padding and resizing.
    """

    # Prepare args

    size = goripy.args.arg_list_to_arg_arr(size, 2).astype(numpy.uint16).tolist()

    fill_value_size = 1 if len(arr.shape) == 2 else arr.shape[2]
    fill_value = goripy.args.arg_list_to_arg_arr(
        fill_value, fill_value_size
    ).astype(arr.dtype)

    # Pad array and fill padding

    arr_h, arr_w = arr.shape[0], arr.shape[1]
    tgt_h, tgt_w = size

    arr_ar = arr_h / arr_w
    tgt_ar = tgt_h / tgt_w

    if arr_ar < tgt_ar:
    
        arr_new_h = round(arr_w / tgt_w * tgt_h)

        pad_top = (arr_new_h - arr_h) // 2
        pad_bot = arr_new_h - arr_h - pad_top

        pad_width = ((pad_top, pad_bot), (0, 0), (0, 0))

        arr = numpy.pad(arr, pad_width=pad_width)

        arr[:pad_top, :, :] = fill_value[None, None, :] 
        arr[arr_new_h-pad_bot:, :, :] = fill_value[None, None, :]

    else:

        arr_new_w = round(arr_h / tgt_h * tgt_w)

        pad_left = (arr_new_w - arr_w) // 2
        pad_right = arr_new_w - arr_w - pad_left

        pad_width = ((0, 0), (pad_left, pad_right), (0, 0))

        arr = numpy.pad(arr, pad_width=pad_width)

        arr[:, :pad_left, :] = fill_value[None, None, :] 
        arr[:, arr_new_w-pad_right:, :] = fill_value[None, None, :]

    # Resize array

    arr = albumentations.geometric.functional.resize(arr, size, cv2.INTER_LINEAR)    

    return arr



def to_rgb(
    arr
):
    """
    Converts an image to RGB.
    
    Requires input data type `numpy.uint8`.
    Output data type will also be `numpy.uint8`.

    Can handle the following shapes:
      - (H x W): Grayscale with no channel dimension.
      - (H x W x 1): Grayscale with channel dimension.
      - (H x W x 4): RGBA.

    Args:

        arr (numpy.ndarray):
            Original array.

    Returns:
    
        numpy.ndarray:
            New array with (H x W x 3) shape.
    """

    if len(arr.shape) == 3:

        if arr.shape[2] == 1:
        
            arr = skimage.color.gray2rgb(arr[:, :, 0])
            return arr

        elif arr.shape[2] == 3:
        
            return arr

        elif arr.shape[2] == 4:
        
            arr = skimage.color.rgba2rgb(arr)
            arr = skimage.util.img_as_ubyte(arr)
            return arr
        
    elif len(arr.shape) == 2:

        arr = skimage.color.gray2rgb(arr)
        return arr

    raise ValueError("Unexpected array shape: {:s}".format(str(arr.shape)))
