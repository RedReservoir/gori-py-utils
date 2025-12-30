import math
import random

import numpy

import skimage
import cv2

import goripy.args
import goripy.mask.bbox



class RandomMaskShadower:
    """
    Randomly casts a shadow on an image using a mask.

    Args:

        shd_round_factors (2-tuple of float or float):
            Shadow round factor range (wrt. image size).
            The bigger this value, the more round the shadow will be.
            Must be a subset of the [0.0, 1.0] range.
            Minimum / maximum recommended range: [0.01, 0.05]
            If one float is provided, this value will be fixed.

        shd_blur_sizes (2-tuple of int or int):
            Shadow gaussian blurring kernel size range.
            The smaller this value, the sharper the shadow will be.
            Must be a pair of odd integers greater or equal than 1.
            No differences can be appreciated with values greater or equal than 21.
            If one integer is provided, this value will be fixed.

        shd_alphas (2-tuple of float or float):
            Shadow intensity alpha parameter range.
            Must be a subset of the [0.0, 1.0] range.
            If one value is provided, this value will be fixed.
            
        shd_shear_coefs (2-tuple of float or float):
            Shadow shearing coefficient range (wrt. shadow size).
            Affects the shape of the shadow.
            Must be a subset of the [-1.0, 1.0] range.

        shd_shear_coefs (2-tuple of float or float):
            Shadow translation coefficient range (wrt. shadow size).
            Affects the position of the shadow.
            Must be a subset of the [-1.0, 1.0] range.
    """

    def __init__(
        self,
        shd_round_factors,
        shd_blur_sizes,
        shd_alphas,
        shd_shear_coefs,
        shd_trans_coefs
    ):
        
        self._shd_round_factors = goripy.args.arg_list_to_arg_arr(
            shd_round_factors, 2
        ).astype(float).tolist()

        self._shd_blur_sizes = goripy.args.arg_list_to_arg_arr(
            shd_blur_sizes, 2
        ).astype(int).tolist()

        self._shd_alphas = goripy.args.arg_list_to_arg_arr(
            shd_alphas, 2
        ).astype(float).tolist()
        
        self._shd_shear_coefs = goripy.args.arg_list_to_arg_arr(
            shd_shear_coefs, 2
        ).astype(float).tolist()
        
        self._shd_trans_coefs = goripy.args.arg_list_to_arg_arr(
            shd_trans_coefs, 2
        ).astype(float).tolist()

        self.randomize()


    def randomize(
        self
    ):
        """
        Randomizes internal values of this transformation.
        Called once automatically on creation.
        """
        
        self._shd_round_factor = random.uniform(*(self._shd_round_factors))
        self._shd_blur_size = random.choice(range(self._shd_blur_sizes[0], self._shd_blur_sizes[1] + 1, 2))
        self._shd_alpha = random.uniform(*(self._shd_alphas))

        self._shd_shear_coef_arr = numpy.random.uniform(*(self._shd_shear_coefs), size=(4, 1))
        self._shd_shear_trans_arr = numpy.random.uniform(*(self._shd_trans_coefs), size=(1, 2))
        

    def __call__(
        self,
        img,
        mask,
        mask_top=None
    ):
        """
        Applies this transformation.
        
        Args:

            img (numpy.ndarray):
                Image to cast the shadow to.
                Must have shape (H x W x 3).

            mask (numpy.ndarray):
                Boolean mask of the object in the image that casts a shadow.
                Must have shape (H x W) and `bool` dtype.

            top_mask (numpy.ndarray, optional):
                Boolean mask of other objects on top of the main object.
                Must have shape (H x W) and `bool` dtype.
                Ignored if not provided.
        """

        return cast_shadow(
            img,
            mask,
            self._generate_random_shd_persp_mat(mask),
            mask_top,
            self._shd_round_factor,
            self._shd_blur_size,
            self._shd_alpha
        )


    def _generate_random_shd_persp_mat(
        self,
        mask
    ):
        """
        Generates a perspective matrix to use with the `cast_shadow` method.
        Based on the `generate_random_shd_persp_mat` method from this module.

        Args:

            mask (numpy.ndarray):
                Mask of the object in the image that casts a shadow.
                Must have shape (H x W) and `bool` dtype.
        """
        
        # Compute mask dimensions

        mask_x0, mask_y0, mask_x1, mask_y1 = goripy.mask.bbox.mask_to_bbox(mask)
        mask_h, mask_w = mask_y1 - mask_y0, mask_x1 - mask_x0 

        # Original points

        old_pts = numpy.float32([
            [mask_y0, mask_x0],
            [mask_y1, mask_x0],
            [mask_y0, mask_x1],
            [mask_y1, mask_x1]
        ])

        new_pts = old_pts.copy()

        # Shearing

        shear_vecs = numpy.float32([
            [-mask_h, -mask_w],
            [mask_h, -mask_w],
            [-mask_h, mask_w],
            [mask_h, mask_w]
        ])

        new_pts += shear_vecs * self._shd_shear_coef_arr / 2

        # Translation

        trans_size = math.sqrt(mask_h * mask_w)
        new_pts += trans_size * self._shd_shear_trans_arr / 2

        # Generate perspective matrix

        persp_mat = cv2.getPerspectiveTransform(old_pts, new_pts)
        return persp_mat



def cast_shadow(
    img,
    mask,
    shd_persp_mat,
    top_mask=None,
    shd_round_factor=0.01,
    shd_blur_size=5,
    shd_alpha=0.25
):
    """
    Applies a shadow to an object in an image.
    Also applies a random perspective transform to the shadow beforehand.

    Args:
    
        img (numpy.ndarray):
            Image to cast the shadow to.
            Must have shape (H x W x 3).

        mask (numpy.ndarray):
            Boolean mask of the object in the image that casts a shadow.
            Must have shape (H x W) and `bool` dtype.

        sdw_persp_mat (numpy.ndarray):
            Perspective matrix to use with cv2.warpPerspective.
            It is recommended to create this matrix via the `cv2.getPerspectiveTransform` method or
            the `generate_random_shd_persp_mat` method provided in this module.

        top_mask (numpy.ndarray, optional):
            Boolean mask of other objects on top of the main object.
            Must have shape (H x W) and `bool` dtype.
            Ignored if not provided.

        shd_round_factor (float, optional):
            Shadow round factor (wrt. image size).
            The bigger this value, the more round the shadow will be.
            Must be a float in the [0.0, 1.0] range.
            It is recommended to set this to 0.05 maximum.
            Defaults to 0.01.

        shd_blur_size (int, optional):
            Shadow gaussian blurring kernel size.
            The smaller this value, the sharper the shadow will be.
            Must be an odd integer greater or equal than 1.
            No differences can be appreciated with values greater or equal than 21.
            Defaults to 5.

        shd_alpha (float, optional):
            Shadow intensity alpha parameter.
            Must be a number in the [0.0, 1.0] range.
            Defaults to 0.25.

    Returns:

        numpy.ndarray:
            Image with the cast shadow.
    """

    # Get image sizes

    img_h = img.shape[0]
    img_w = img.shape[1]

    img_s = round(math.sqrt(img_h * img_w))

    # Create shadow image

    shd_bw_img = skimage.util.img_as_float(mask)

    # Use dilation-erosion to make shadow round

    round_ker_size = round(shd_round_factor * img_s)

    shd_bw_img = cv2.dilate(
        shd_bw_img,
        kernel=skimage.morphology.disk(round_ker_size)
    )
    
    shd_bw_img = cv2.erode(
        shd_bw_img,
        kernel=skimage.morphology.footprint_rectangle((round_ker_size * 2, round_ker_size * 2))
    )
    
    # Use gaussian blur to control shadow intensity

    shd_bw_img = cv2.GaussianBlur(
        shd_bw_img,
        ksize=(shd_blur_size, shd_blur_size),
        sigmaX=0,
        sigmaY=0
    )

    # Apply perspective transform

    shd_bw_img = cv2.warpPerspective(
        shd_bw_img,
        shd_persp_mat,
        (img_w, img_h)
    )

    # Compute shadow mask with objects

    shd_bw_img *= shd_alpha

    shd_mask = ~mask
    if top_mask is not None:
        shd_mask &= ~top_mask

    # Apply shadow to image

    img = skimage.util.img_as_float(img)
    img *= 1 - (shd_bw_img[:, :, None] * (shd_mask[:, :, None]))
    img = skimage.util.img_as_ubyte(img)

    return img



def generate_random_shd_persp_mat(
    mask,
    shd_shear_coefs,
    shd_trans_coefs
):
    """
    Randomly generates a perspective matrix to use with the `cast_shadow` method.
    
    To generate the matrix, 4-point transformation outputs are given. The initial points are
    the vertices of the mask box. Shearing and translation are applied beforehand.

    Shearing is performed by moving the vertices further or closer to the center.
    Translation is performed by simply applying a translation operation to the vertices.

    Args:

        mask (numpy.ndarray):
            Mask of the object in the image that casts a shadow.
            Must have shape (H x W) and `bool` dtype.

        shd_shear_coefs (2-tuple of float):
            Shearing coefficient range (wrt. shadow size).
            Affects the shape of the shadow.
            Shearing coefficients are picked randomly from this range.
            Must be a subset of the [-1.0, 1.0] range.

        shd_trans_coefs (2-tuple of float):
            Translation coefficient range (wrt. shadow size).
            Affects the position of the shadow.
            Translation coefficients are picked randomly from this range.
            Must be a subset of the [-1.0, 1.0] range.
    """

    # Prepare args

    shd_shear_coefs = goripy.args.arg_list_to_arg_arr(
        shd_shear_coefs, 2
    ).astype(float).tolist()
    
    shd_trans_coefs = goripy.args.arg_list_to_arg_arr(
        shd_trans_coefs, 2
    ).astype(float).tolist()

    # Compute mask dimensions

    mask_x0, mask_y0, mask_x1, mask_y1 = goripy.mask.bbox.mask_to_bbox(mask)
    mask_h, mask_w = mask_y1 - mask_y0, mask_x1 - mask_x0 

    # Original points

    old_pts = numpy.asarray([
        [mask_y0, mask_x0],
        [mask_y1, mask_x0],
        [mask_y0, mask_x1],
        [mask_y1, mask_x1]
    ], dtype=float)

    new_pts = old_pts.copy()

    # Shearing

    shear_vecs = numpy.asarray([
        [-mask_h, -mask_w],
        [mask_h, -mask_w],
        [-mask_h, mask_w],
        [mask_h, mask_w]
    ], dtype=float)

    shear_coefs = numpy.random.uniform(*(shd_shear_coefs), size=(4, 1)).astype(numpy.float32)
    new_pts += shear_vecs * shear_coefs / 2

    # Translation

    trans_size = math.sqrt(mask_h * mask_w)
    trans_coefs = numpy.random.uniform(*(shd_trans_coefs), size=(1, 2)).astype(numpy.float32)
    new_pts += trans_size * trans_coefs / 2

    # Generate perspective matrix

    persp_mat = cv2.getPerspectiveTransform(old_pts, new_pts)
    return persp_mat
