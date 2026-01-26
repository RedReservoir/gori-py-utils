import numpy

import goripy.mask.bbox
import goripy.args



class SquareMaskCropper:
    """
    Crops arrays based on a reference mask.

    To fit this object to a particular mask, call the `fit` method.
    Later, to crop other arrays, call the `crop` method.
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
        Fits this object using a reference mask, to later crop other arrays.

        Args:
        
            mask (numpy.ndarray):
                Mask to use as reference for cropping.
                Must have shape (H x W) and `bool` dtype.

            border_ratio (float, optional):
                Percentage of border with respect to the crop contents.
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
        arr,
        bkg_value=0
    ):
        """
        Crops an array.

        Args:

            arr (numpy.ndarray):
                Numpy array to crop.
                Must have shape (H x W) or (H x W x C).

            bkg_value (any, optional):
                Values to set as background outside of the cropping area.
                Defaults to 0.

        Returns:

            numpy.ndarray:
                The cropped array.
        """

        if not self._is_fit:
            raise ValueError("Cannot call crop method before calling fit method")

        bkg_value_size = 1 if len(arr.shape) == 2 else arr.shape[2]
        bkg_value = goripy.args.arg_list_to_arg_arr(
            bkg_value, bkg_value_size, arr.dtype
        )

        # Create crop array

        if len(arr.shape) == 2:
            
            crop_arr = numpy.empty(
                shape=(self._new_size, self._new_size),
                dtype=arr.dtype
            )

        else:

            crop_arr = numpy.empty(
                shape=(self._new_size, self._new_size, arr.shape[2]),
                dtype=arr.dtype
            )
            
        # Copy cropped area

        crop_arr[self._new_copy_y0:self._new_copy_y1, self._new_copy_x0:self._new_copy_x1] =\
            arr[self._orig_copy_y0:self._orig_copy_y1, self._orig_copy_x0:self._orig_copy_x1]

        # Apply background to area outside crop and border

        crop_arr[:self._new_copy_y0, :] = bkg_value
        crop_arr[self._new_copy_y1:, :] = bkg_value
        crop_arr[:, :self._new_copy_x0] = bkg_value
        crop_arr[:, self._new_copy_x1:] = bkg_value

        return crop_arr
