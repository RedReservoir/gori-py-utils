import xml

import numpy

import cv2

import goripy.mask.bbox
import goripy.mask.rle



def read_mask_from_xml_item(
    mask_xml_item,
    img_h,
    img_w,
):
    """
    Reads a segmentation mask from a CVAT format annotation file.
    The following fields of `mask_xml_item` are read:

    - `rle`
    - `top`
    - `left`
    - `height`
    - `width`

    Args:
        mask_xml_item (xml.etree.ElementTree.ElementTree):
            Target `<mask>` XML node to read the mask from.
        img_h (int):
            Original image height.
            Can be found in the parent `<image>` XML node.
        img_w (int):
            Original image width.
            Can be found in the parent `<image>` XML node.

    Returns:
        numpy.ndarray:
            A 2D boolean numpy array with the mask.
    """

    mask_rle_arr = numpy.fromiter(map(int, mask_xml_item.get("rle").split(", ")), dtype=int)
    
    mask_x = round(float(mask_xml_item.get("left")))
    mask_y = round(float(mask_xml_item.get("top")))
    mask_w = round(float(mask_xml_item.get("width")))
    mask_h = round(float(mask_xml_item.get("height")))

    mask = goripy.mask.rle.rle_to_mask(mask_rle_arr, (mask_w, mask_h)).T
    
    full_mask = numpy.zeros(shape=(img_h, img_w), dtype=bool)
    full_mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = mask[:, :]

    return full_mask



def read_polygon_from_xml_item(
    poly_xml_item,
    img_h,
    img_w,
):
    """
    Reads a polygon mask from a CVAT format annotation file.
    The following fields of `poly_xml_item` are read:

    - `points`

    Args:
        poly_xml_item (xml.etree.ElementTree.ElementTree):
            Target `<polygon>` XML node to read the mask from.
        img_h (int):
            Original image height.
            Can be found in the parent `<image>` XML node.
        img_w (int):
            Original image width.
            Can be found in the parent `<image>` XML node.

    Returns:
        numpy.ndarray:
            A 2D boolean numpy array with the mask.
    """

    poly_point_arrr = numpy.round(numpy.asarray([
        list(map(float, point_str.split(",")))
        for point_str in poly_xml_item.get("points").split(";")
    ])).astype(numpy.int32)

    mask = numpy.zeros(shape=(img_h, img_w), dtype=numpy.uint8)
    mask = cv2.fillPoly(mask, pts=[poly_point_arrr], color=1)
    
    return mask



def write_mask_to_xml_item(
    mask_xml_item,
    mask
):
    """
    Writes a segmentation mask to a CVAT format annotation file.
    The following fields of `mask_xml_item` are modified:

    - `rle`
    - `top`
    - `left`
    - `height`
    - `width`

    Args:
        mask_xml_item (xml.etree.ElementTree.ElementTree):
            Target `<mask>` XML node to write the mask to.
        mask (numpy.ndarray):
            A 2D boolean numpy array with the mask.
    """

    mask = mask.copy()
    mask = mask.T
    
    y0, x0, y1, x1 = goripy.mask.bbox.mask_to_bbox(mask) 
    mask = mask[x0:x1, y0:y1]
    
    mask_rle_arr = goripy.mask.rle.mask_to_rle(mask)
    mask_rle_str = ", ".join([str(rle_num) for rle_num in mask_rle_arr])

    mask_xml_item.set("rle", mask_rle_str)
    
    mask_xml_item.set("left", str(x0))
    mask_xml_item.set("top", str(y0))
    mask_xml_item.set("width", str(x1 - x0))
    mask_xml_item.set("height", str(y1 - y0))
