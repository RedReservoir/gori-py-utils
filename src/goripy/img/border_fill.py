import random
import numpy

import goripy.args



class RandomBorderFiller:
    """
    Randomly fills a an array center or border.

    Args:

        top_brs (2-tuple of float or float):
            Min. / max. percentage of top border to fill / not fill.
            If one value is provided, this value will be fixed.
        bot_brs (2-tuple of float or float):
            Min. / max. percentage of bottom border to fill / not fill.
            If one value is provided, this value will be fixed.
        left_brs (2-tuple of float or float):
            Min. / max. percentage of left border to fill / not fill.
            If one value is provided, this value will be fixed.
        right_brs (2-tuple of float or float):
            Min. / max. percentage of right border to fill / not fill.
            If one value is provided, this value will be fixed.

        fill_border (bool):
            If True, the borders of the array will be filled out.
            If False, the center of the array will be filled out instead.
        fill_value (any):
            Value to fill the array with. Must be compatible with the array.
    """

    def __init__(
        self,
        top_brs,
        bot_brs,
        left_brs,
        right_brs,   
        fill_border,
        fill_value
    ):

        self._top_brs = goripy.args.arg_list_to_arg_arr(
            top_brs, 2, float
        ).tolist()

        self._bot_brs = goripy.args.arg_list_to_arg_arr(
            bot_brs, 2, float
        ).tolist()

        self._left_brs = goripy.args.arg_list_to_arg_arr(
            left_brs, 2, float
        ).tolist()

        self._right_brs = goripy.args.arg_list_to_arg_arr(
            right_brs, 2, float
        ).tolist()

        self._fill_border = fill_border
        self._fill_value = fill_value

        self.randomize()


    def randomize(
        self
    ):
        """
        Randomizes internal values of this transformation.
        Called once automatically on creation.
        """

        self._top_br = random.uniform(*(self._top_brs))
        self._bot_br = random.uniform(*(self._bot_brs))
        self._left_br = random.uniform(*(self._left_brs))
        self._right_br = random.uniform(*(self._right_brs))


    def __call__(
        self,
        arr,
        fill_border=None,
        fill_value=None
    ):
        """
        Applies this transformation.
        This operation is performed inplace
        
        Args:

            arr (numpy.ndarray):
                Numpy array to fill the border or center of.
                Must have shape (H x W) or (H x W x C).
            
            fill_border (bool, optional):
                Override of constructor and randomized values.
                If True, the borders of the array will be filled out.
                If False, the center of the array will be filled out instead.
            fill_value (any, optional):
                Override of constructor and randomized values.
            Value to fill the array with. Must be compatible with the array.
        """

        fill_border_rel(
            arr,
            self._top_br,
            self._bot_br,
            self._left_br,
            self._right_br,
            self._fill_border if fill_border is None else fill_border,
            self._fill_value if fill_value is None else fill_value
        )



def fill_border_abs(
    arr,
    top_b,
    bot_b,
    left_b,
    right_b,
    fill_border,
    fill_value
):
    """
    Fills the border or center of an array.
    This operation is performed inplace.

    Args:

        arr (numpy.ndarray):
            Numpy array to fill the border or center of.
            Must have shape (H x W) or (H x W x C).
            
        top_b (int):
            Amount of top border to fill / not fill.
        bot_b (int):
            Amount of bottom border to fill / not fill.
        left_b (int):
            Amount of left border to fill / not fill.
        right_b (int):
            Amount of right border to fill / not fill.

        fill_border (bool):
            If True, the borders of the array will be filled out.
            If False, the center of the array will be filled out instead.
        fill_value (any):
            Value to fill the array with. Must be compatible with the array.
    """

    fill_value_size = 1 if len(arr.shape) == 2 else arr.shape[2]
    fill_value = goripy.args.arg_list_to_arg_arr(
        fill_value, fill_value_size, arr.dtype
    )

    arr_h = arr.shape[0]
    arr_w = arr.shape[1]

    cen_y0 = top_b
    cen_y1 = arr_h - bot_b
    cen_x0 = left_b
    cen_x1 = arr_w - right_b

    if fill_border:
        arr[:cen_y0, :] = fill_value
        arr[cen_y1:, :] = fill_value
        arr[:, :cen_x0] = fill_value
        arr[:, cen_x1:] = fill_value
    else:
        arr[cen_y0:cen_y1, cen_x0:cen_x1] = fill_value

    

def fill_border_rel(
    arr,
    top_br,
    bot_br,
    left_br,
    right_br,
    fill_border,
    fill_value
):
    """
    Fills the border or center of an array.
    This operation is performed inplace.

    Args:

        arr (numpy.ndarray):
            Numpy array to fill the border or center of.
            Must have shape (H x W) or (H x W x C).

        top_br (float):
            Percentage of top border to fill / not fill.
            Must lie in the [0.0, 1.0] interval.
        bot_br (float):
            Percentage of bottom border to fill / not fill.
            Must lie in the [0.0, 1.0] interval.
        left_br (float):
            Percentage of left border to fill / not fill.
            Must lie in the [0.0, 1.0] interval.
        right_br (float):
            Percentage of right border to fill / not fill.
            Must lie in the [0.0, 1.0] interval.

        fill_border (bool):
            If True, the borders of the array will be filled out.
            If False, the center of the array will be filled out instead.
        fill_value (any):
            Value to fill the array with. Must be compatible with the array.
    """

    fill_value_size = 1 if len(arr.shape) == 2 else arr.shape[2]
    fill_value = goripy.args.arg_list_to_arg_arr(
        fill_value, fill_value_size, arr.dtype
    )

    arr_h = arr.shape[0]
    arr_w = arr.shape[1]
    
    cen_y0 = round(arr_h * (top_br / 2))
    cen_y1 = round(arr_h * (1 - (bot_br / 2)))
    cen_x0 = round(arr_w * (left_br / 2))
    cen_x1 = round(arr_w * (1 - (right_br / 2)))

    if fill_border:
        arr[:cen_y0, :] = fill_value
        arr[cen_y1:, :] = fill_value
        arr[:, :cen_x0] = fill_value
        arr[:, cen_x1:] = fill_value
    else:
        arr[cen_y0:cen_y1, cen_x0:cen_x1] = fill_value
