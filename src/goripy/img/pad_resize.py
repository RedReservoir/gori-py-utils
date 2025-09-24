import numpy

import albumentations
import cv2

import goripy.args



def pad_resize_image(
    img,
    size,
    fill=255
):
    """
    Pads and then resizes an image.

    :param img: numpy.ndarray
        Image (H x W x C or H x W) (numpy.uint8) to pad and resize.
    :param size: int or list of int or tuple of int
        The target height and width of the image.
        If one value is provided, both height and width will be set to that value.
    :param fill: int or list of int or tuple of int, default=255
        RGB channel values to fill the image with padding.
        If one value is provided, all channels will be set to that value.

    :return: numpy.ndarray
        Image (H x W x C or H x W) (numpy.uint8) after padding and resizing.
    """

    # Prepare args

    size = goripy.args.arg_list_to_arg_arr(size, 2).astype(numpy.uint16).tolist()
    fill_arr = goripy.args.arg_list_to_arg_arr(fill, 3).astype(numpy.uint8)

    # Pad and fill image with color

    img_h, img_w = img.shape[0], img.shape[1]
    tgt_h, tgt_w = size

    img_ar = img_h / img_w
    tgt_ar = tgt_h / tgt_w

    if img_ar < tgt_ar:
    
        img_new_h = round(img_w / tgt_w * tgt_h)

        pad_top = (img_new_h - img_h) // 2
        pad_bot = img_new_h - img_h - pad_top

        pad_width = ((pad_top, pad_bot), (0, 0), (0, 0))

        img = numpy.pad(img, pad_width=pad_width)

        img[:pad_top, :, :] = fill_arr[None, None, :] 
        img[img_new_h-pad_bot:, :, :] = fill_arr[None, None, :]

    else:

        img_new_w = round(img_h / tgt_h * tgt_w)

        pad_left = (img_new_w - img_w) // 2
        pad_right = img_new_w - img_w - pad_left

        pad_width = ((0, 0), (pad_left, pad_right), (0, 0))

        img = numpy.pad(img, pad_width=pad_width)

        img[:, :pad_left, :] = fill_arr[None, None, :] 
        img[:, img_new_w-pad_right:, :] = fill_arr[None, None, :]

    # Resize image

    img = albumentations.geometric.functional.resize(img, size, cv2.INTER_LINEAR)    

    return img
