import numpy

import goripy.mask.bbox
import goripy.args



class SquareMaskCropper:
    """
    Implements cropping of images based on a reference mask.
    Implementation for square crops with padding.
    """

    def __init__(
        self
    ):

        self._is_fit = False


    def fit(
        self,
        mask,
        border_ratio=0.0,
        min_size=128,
    ):
        """
        Fits this object using a reference mask, to later crop other images and masks.

        Args:
        
            mask (numpy.ndarray):
                2D numpy array with `bool` dtype to use as reference for cropping.

            border_ratio (float, optional):
                Percentage of border wrt. to the crop contents.
                Must be a float in the [0, 1) interval.
                Defaults to 0.0.

            min_size (int, optional):
                Minimum original image height and width to crop.
                Defaults to 128.
        """

        # Get original size and bbox limits

        orig_h, orig_w = mask.shape
        mask_x0, mask_y0, mask_x1, mask_y1 = goripy.mask.bbox.mask_to_bbox(mask)

        # Compute mask center and size

        mask_yc = (mask_y0 + mask_y1) / 2
        mask_xc = (mask_x0 + mask_x1) / 2

        mask_h = mask_y1 - mask_y0
        mask_w = mask_x1 - mask_x0

        # Compute resize ratio and new size

        exp_ratio = 1 / (1 - border_ratio)
        new_size = round(max(mask_h, mask_w) * exp_ratio)
        new_size = max(new_size, min_size)

        self._new_size = new_size

        # Compute original cropping limits (may be outside range)

        orig_crop_y0 = round(mask_yc - (new_size / 2))
        orig_crop_y1 = orig_crop_y0 + new_size

        orig_crop_x0 = round(mask_xc - (new_size / 2))
        orig_crop_x1 = orig_crop_x0 + new_size

        # Compute spillout border sizes wrt crops

        new_top_border = max(0 - orig_crop_y0, 0)
        new_bottom_border = max(orig_crop_y1 - orig_h, 0)

        new_left_border = max(0 - orig_crop_x0, 0)
        new_right_border = max(orig_crop_x1 - orig_w, 0)

        # Compute copy and border limits

        self._orig_copy_y0 = max(orig_crop_y0, 0)
        self._orig_copy_y1 = min(orig_crop_y1, orig_h)

        self._orig_copy_x0 = max(orig_crop_x0, 0)
        self._orig_copy_x1 = min(orig_crop_x1, orig_w)

        self._new_copy_y0 = new_top_border
        self._new_copy_y1 = new_size - new_bottom_border
        
        self._new_copy_x0 = new_left_border
        self._new_copy_x1 = new_size - new_right_border

        # Activate

        self._is_fit = True


    def crop(
        self,
        img,
        bkg=0
    ):
        """
        Crops an image.

        Args:

            img (numpy.ndarray):
                Image to crop with the mask.

            bkg (any, optional):
                Values to set as background.
                Defaults to 0.

        Returns:

            numpy.ndarray:
                The cropped image.
        """

        if not self._is_fit:
            raise ValueError("Cannot call SquareMaskCropper.crop() before calling SquareMaskCropper.fit()")

        bkg_arr = goripy.args.arg_list_to_arg_arr(bkg, 1 if len(img.shape) == 2 else img.shape[2]).astype(img.dtype)

        if len(img.shape) == 2: img_crop = numpy.empty(shape=(self._new_size, self._new_size), dtype=img.dtype)
        else: img_crop = numpy.empty(shape=(self._new_size, self._new_size, img.shape[2]), dtype=img.dtype)
            
        img_crop[self._new_copy_y0:self._new_copy_y1, self._new_copy_x0:self._new_copy_x1] =\
            img[self._orig_copy_y0:self._orig_copy_y1, self._orig_copy_x0:self._orig_copy_x1]

        img_crop[:self._new_copy_y0, :] = bkg_arr
        img_crop[self._new_copy_y1:, :] = bkg_arr
        img_crop[:, :self._new_copy_x0] = bkg_arr
        img_crop[:, self._new_copy_x1:] = bkg_arr

        return img_crop
        