import xml

import numpy

import sklearn
import sklearn.cluster



def compute_box_coords(
    num_cols,
    num_rows,
    plot_size,
    limits,
    img_size,
    sep_size,
    box_h
):
    """
    Generates CVAT box coordinates for a 2D image grid plot.
    The `limits`, `img_size`, `sep_size` and `box_h` args are not expected in pixel-perfect
    precision. Instead, they will be used as relative sizes.
    
    Args:
        num_cols (int):
            Number of columns in the plot.
        num_rows (int):
            Number of rows in the plot.
        plot_size (tuple of int):
            Size of the plot in pixels (height, width).

        limits (tuple of float):
            Outer limits (top, left, bot, right).
        img_size (tuple of float):
            Size of the plot images (height, width).
        sep_size (tuple of float):
            Size of the separation between plot images (height, width).
        box_h (float):
            Height of boxes in CVAT.

    Returns:
        tuple of numpy.ndarray
            A tuple containing:
            
            - ytl_arr: Array with all ytl positions of boxes.
            - ybr_arr: Array with all ybr positions of boxes.
            - xtl_arr: Array with all xtl positions of boxes.
            - xbr_arr: Array with all xbr positions of boxes.
    """

    plot_h, plot_w = plot_size
    top, left, bot, right = limits
    img_h, img_w = img_size
    sep_h, sep_w = sep_size

    ytl_arr = top + numpy.arange(num_rows) * (box_h + img_h + sep_h)
    ybr_arr = ytl_arr + box_h

    xtl_arr = left + numpy.arange(num_cols) * (img_w + sep_w)
    xbr_arr = xtl_arr + img_w

    total_h = top + num_rows * (box_h + img_h) + (num_rows - 1) * sep_h + bot
    total_w = left + num_cols * img_w + (num_cols - 1) * sep_w + right

    ytl_arr = ytl_arr.astype(float)
    ybr_arr = ybr_arr.astype(float)

    xtl_arr = xtl_arr.astype(float)
    xbr_arr = xbr_arr.astype(float)

    ytl_arr *= plot_h / total_h
    ybr_arr *= plot_h / total_h

    xtl_arr *= plot_w / total_w
    xbr_arr *= plot_w / total_w

    ytl_arr = numpy.round(ytl_arr).astype(int)
    ybr_arr = numpy.round(ybr_arr).astype(int)

    xtl_arr = numpy.round(xtl_arr).astype(int)
    xbr_arr = numpy.round(xbr_arr).astype(int)

    ytl_arr = numpy.repeat(ytl_arr, num_cols)
    ybr_arr = numpy.repeat(ybr_arr, num_cols)

    xtl_arr = numpy.tile(xtl_arr, num_rows)
    xbr_arr = numpy.tile(xbr_arr, num_rows)

    return ytl_arr, ybr_arr, xtl_arr, xbr_arr



def get_box_ord_idxs(
    num_cols,
    num_rows,
    box_xml_item_list
):
    """
    Computes ordered <box> XML node indices from a 2D grid based on box centers.

    Args:
        num_cols (int):
            Number of columns formed by the boxes in CVAT.
        num_rows (int):
            Number of rows formed by the boxes in CVAT.
        box_xml_item_list (list of xml.etree.ElementTree.ElementTree):
            List of `<box>` XML nodes to order based on box position.

    Returns:
        numpy.ndarray:
            A 1D numpy array with the rank of each box.
            Iterate the list of `<box>` XML nodes using this array of indices.
    """

    box_cen_arr = numpy.empty(shape=(len(box_xml_item_list), 2), dtype=float)
    for box_xml_item_idx, box_xml_item in enumerate(box_xml_item_list):

        box_cen_arr[box_xml_item_idx, 0] = float(box_xml_item.get("xtl"))
        box_cen_arr[box_xml_item_idx, 1] = float(box_xml_item.get("ytl"))

    #

    kmeans_x_obj = sklearn.cluster.KMeans(n_clusters=num_cols)
    box_x_lab_arr = kmeans_x_obj.fit_predict(box_cen_arr[:, 0:1])
    box_x_clst_arr = kmeans_x_obj.cluster_centers_[:, 0]

    kmeans_y_obj = sklearn.cluster.KMeans(n_clusters=num_rows)
    box_y_lab_arr = kmeans_y_obj.fit_predict(box_cen_arr[:, 1:2])
    box_y_clst_arr = kmeans_y_obj.cluster_centers_[:, 0]

    #

    perm_x_arr = numpy.empty(shape=(num_cols), dtype=int)
    perm_x_arr[numpy.argsort(box_x_clst_arr)] = numpy.arange(num_cols)
    box_x_lab_arr = perm_x_arr[box_x_lab_arr]

    perm_y_arr = numpy.empty(shape=(num_rows), dtype=int)
    perm_y_arr[numpy.argsort(box_y_clst_arr)] = numpy.arange(num_rows)
    box_y_lab_arr = perm_y_arr[box_y_lab_arr]

    #

    box_lab_arr = box_x_lab_arr + box_y_lab_arr * num_cols
    box_ord_idx_arr = numpy.argsort(box_lab_arr)
    
    return box_ord_idx_arr
