import enum
import math

import numpy
import torch

import matplotlib
import matplotlib.pyplot

import goripy.plot.grid



class ShowImgsTitleMode(enum.Enum):

    NO_TITLE = 0
    CUSTOM = 1
    IMAGE_INFO = 2


class ShowImgsGridMode(enum.Enum):

    ONE_ROW = 0
    FIXED_ROWS = 1
    FIXED_COLS = 2
    ADAPTIVE = 3


def show_imgs(
    img_list,
    title_mode=ShowImgsTitleMode.NO_TITLE,
    title_list=None,
    title_size=10,
    suptitle=None,
    grid_mode=ShowImgsGridMode.ONE_ROW,
    num_rows=1,
    num_cols=1,
    wh_ratio=1.0,
    plt_size_w=5,
    plt_size_h=5,
    img_ticks=True,
    plot_filename=None,
    dpi=100
):

    """
    Quickly generate a plot that shows multiple images and their data.

    Args:
    
        img_list (list of numpy.ndarray):
            List of images to show.

        title_mode (ShowImgsTitleMode):
            Options for showing titles on top of images.
            
            - NO_TITLE: No titles will be shown.
            - CUSTOM: Custom titles passed through the title_list argument will be shown.
            - IMAGE_INFO: Image shape, data type and range will be shown.

        title_list (list of str, optional):
            Titles to show on top of images.
            Sometimes ignored or overwritten depending on the title_mode argument.

        title_size (int, optional):
            Title size on top of images. Defaults to 10.
            Sometimes ignored or overwritten depending on the title_mode argument.

        suptitle (str, optional):
            Title to show on top of the plot. If not provided, no title will be shown.

        grid_mode (ShowImgsGridMode):
            Options for controlling the image grid.

            - ONE_ROW: One row of images will be shown.
            - FIXED_ROWS: A custom number of rows passed through the num_rows argument will be shown.
            - ADAPTIVE: Adaptive grid distribution based on the wh_ratio, plt_size_w, plt_size_h arguments.

        num_rows (int, optional):
            Number of rows in the image grid.
            Sometimes ignored or overwritten depending on the grid_mode argument.
            Defaults to 1.

        num_cols (int, optional):
            Number of columns in the image grid.
            Sometimes ignored or overwritten depending on the grid_mode argument.
            Defaults to 1.

        wh_ratio (float, optional):
            Width-to-height ratio in the image grid.
            Sometimes ignored or overwritten depending on the grid_mode argument.
            Defaults to 1.0.

        plt_size_w (float, optional):
            Individual image width.
            Defaults to 5.0.

        plt_size_h (float, optional):
            Individual image height.
            Defaults to 5.0.

        img_ticks (bool, optional):
            If True, shows image size ticks.
            Defaults to True.

        plot_filename (str, optional):
            Filename to which to save the plot image to.
            If not provided, the plot will be shown.

        dpi (int, optional):
            Dots per inch in the plot image.
            Ignored if the plot_filename argument is not provided.
            Defaults to 100.
    
    Returns
    
        dict
            Information about the generated plot.
    """

    # Title modes

    if title_mode == ShowImgsTitleMode.CUSTOM:

        if title_list is None:
            title_list = [""] * len(img_list)

    elif title_mode == ShowImgsTitleMode.NO_TITLE:

        title_list = [""] * len(img_list)

    elif title_mode == ShowImgsTitleMode.IMAGE_INFO:

        title_list = []

        for img in img_list:

            img_shape = str(tuple(img.shape))
            img_dtype = str(img.dtype)
            img_min = str(img.min().item()) if type(img) is torch.Tensor else str(img.min())
            img_max = str(img.max().item()) if type(img) is torch.Tensor else str(img.max())
            
            title = "{:s}\n{:s} [{:s}, {:s}]".format(
                img_shape,
                img_dtype,
                img_min,
                img_max
            )

            title_list.append(title)

    # Grid modes

    num_plots = len(img_list)

    if grid_mode == ShowImgsGridMode.ONE_ROW:

        num_rows = 1
        num_cols = round(math.ceil(num_plots / num_rows))
    
    elif grid_mode == ShowImgsGridMode.FIXED_ROWS:

        num_cols = round(math.ceil(num_plots / num_rows))
    
    elif grid_mode == ShowImgsGridMode.FIXED_COLS:

        num_rows = round(math.ceil(num_plots / num_cols))

    elif grid_mode == ShowImgsGridMode.ADAPTIVE:

        num_cols, num_rows = goripy.plot.grid.compute_best_2d_grid_dims(num_plots, wh_ratio, plt_size_w, plt_size_h)

    # Generate plot    

    fig, axs = matplotlib.pyplot.subplots(ncols=num_cols, nrows=num_rows, figsize=(plt_size_w * num_cols, plt_size_h * num_rows))
    if len(img_list) == 1: axs = numpy.asarray([axs])
    axs = axs.flatten()

    for ax, img, title in zip(axs, img_list, title_list):
        ax.imshow(img)
        ax.set_title(title, fontsize=title_size)
        if not img_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    
    for ax_idx in range(num_plots, num_cols * num_rows):
        axs[ax_idx].axis("off")

    if suptitle is not None:
        matplotlib.pyplot.suptitle(suptitle)

    matplotlib.pyplot.tight_layout()

    if plot_filename is None:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.savefig(plot_filename, bbox_inches="tight", dpi=dpi)
        matplotlib.pyplot.close()

    # Return plot data

    return {
        "num_plots": num_plots,
        "num_cols": num_cols,
        "num_rows": num_rows
    }
