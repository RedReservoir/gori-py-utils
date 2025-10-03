import numpy

import skimage
import cv2



def cast_shadow(
    img,
    mask,
    top_mask=None,
    shadow_dilate_disk_size=3,
    shadow_gauss_blur_size=5,
    shadow_persp_mat=None,
    shadow_alpha=0.2
):
    """
    Applies a shadow to an object in an image.
    Also applies a random perspective transform to the shadow beforehand.

    Args:
    
        img (numpy.ndarray):
            Image to apply the shadow to.

        mask (numpy.ndarray):
            Boolean mask of the object in the image.

        top_mask (numpy.ndarray, optional):
            Boolean mask of other objects on top of the main object.
            Ignored if not provided.

        shadow_dilate_disk_size (int, optional):
            Shadow dilation operation disk footprint size.
            Defaults to 3.

        shadow_gauss_blur_size (int, optional):
            Shadow gaussian blurring kernel size.
            Defaults to 5.

        shadow_persp_mat (numpy.ndarray):
            Perspective matrix to use with cv2.warpPerspective.
            It is recommended to create this matrix via cv2.getPerspectiveTransform.

        shadow_alpha (float, optional):
            Shadow intensity alpha parameter.
            Must be a number in the [0, 1] interval.
            Defaults to 0.2.

    Returns:

        numpy.ndarray:
            Image with the applied shadow.
    """

    # Create shadow mask, dilate and blur

    shadow = mask.astype(numpy.uint8)
    shadow = cv2.dilate(shadow, skimage.morphology.disk(shadow_dilate_disk_size))
    shadow = shadow.astype(float)
    shadow = cv2.GaussianBlur(shadow, (shadow_gauss_blur_size, shadow_gauss_blur_size), 0)

    # Apply perspective transform

    shadow = cv2.warpPerspective(shadow, shadow_persp_mat, (img.shape[1], img.shape[0]))

    # Apply shadow to image

    shadow *= shadow_alpha

    shadow_mask = ~mask
    if top_mask is not None:
        shadow_mask &= ~top_mask

    img = img.astype(float) / 255.0
    img *= 1 - (shadow[:, :, None] * (shadow_mask[:, :, None]))
    img = numpy.round(img * 255.0).astype("uint8")

    return img



def generate_persp_mat(
    self,
    mask,
    min_shear_coef,
    max_shear_coef,
    min_trans_coef,
    max_trans_coef
):
    """
    Randomly generates a perspective matrix to use with the cast_shadow method.
    The matrix uses as basis the mask of the object to cast a shadow from.

    To generate the matrix, 4 point transformation outputs are given. The initial points are
    the vertices of the mask bbox. Shearing and translation are applied beforehand.

    Shearing is performed by moving the vertices further or closer to the center.
    Translation is performed by simply applying a translation operation to the vertices.

    Args:

        mask (numpy.ndarray):
            Mask of the object to cast a shadow from.

        min_shear_coef (float):
            Minimum shearing coefficient (wrt. shadow size).
            Must be a number in the [-1, 1] interval.

        max_shear_coef (float):
            Maximum shearing coefficient (wrt. shadow size).
            Must be a number in the [-1, 1] interval.

        min_trans_coef (float):
            Minimum translation coefficient (wrt. shadow size).
            Must be a number in the [-1, 1] interval.

        max_trans_coef (float):
            Maximum translation coefficient (wrt. shadow size).
            Must be a number in the [-1, 1] interval.
    """

    # Compute mask dimensions

    mask_x0, mask_y0, mask_x1, mask_y1 = src.py_utils.mask.mask_to_bbox(mask)
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

    shear_coefs = numpy.random.uniform(min_shear_coef, max_shear_coef, size=(4, 1)).astype(numpy.float32)

    new_pts += shear_vecs * shear_coefs / 2

    # Translation

    trans_size = math.sqrt(mask_h * mask_w)
    trans_coefs = numpy.random.uniform(min_trans_coef, max_trans_coef, size=(1, 2)).astype(numpy.float32)

    new_pts += trans_size * trans_coefs / 2

    # Generate perspective matrix

    persp_mat = cv2.getPerspectiveTransform(old_pts, new_pts)
    return persp_mat
