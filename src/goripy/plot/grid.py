import math



def compute_best_2d_grid_dims(
    size,
    wh_ratio=1.0,
    plt_w=1.0,
    plt_h=1.0,
    allow_diff_size=True
):
    """
    Computes optimal number of columns and rows in a 2D plot grid.

    Args:

        size (int):
            Number of plots in the 2D plot grid.

        wh_ratio (float, optional):
            Desired width:height ratio in the 2D plot grid.
            Ratio expressed in terms of absolute size.
            Defaults to 1.0.

        plt_w (float, optional):
            Single plot width.
            Defaults to 1.0.

        plt_h (float, optional):
            Single plot height.
            Defaults to 1.0.

        allow_diff_size (bool, optional):
            If True, allows for results with number of plots different than `size`.
            Defaults to True.

    Returns:
    
        tuple of int
            A tuple containing:
            
            - ncols: 2D plot grid number of columns.
            - nrows: 2D plot grid number of rows.
    """

    # Size 1 fallback

    if size == 1:
        return 1, 1

    # Compute list of cols and rows

    ncols, nrows = size, 1
    ncols_nrows_list = [(ncols, nrows)]

    while ncols >= nrows:

        new_ncols = ncols - 1
        new_nrows = round(math.ceil(size / new_ncols))
        new_ncols = round(math.ceil(size / new_nrows))

        if allow_diff_size or (new_ncols * new_nrows) == size:
            ncols_nrows_list.append((new_ncols, new_nrows))
        ncols, nrows = new_ncols, new_nrows

    ncols_nrows_list = ncols_nrows_list[:-1]

    if ncols_nrows_list[-1][0] == ncols_nrows_list[-1][1]:
        ncols_nrows_list += [(num_rows, num_cols) for num_cols, num_rows in ncols_nrows_list[-2::-1]]
    else:
        ncols_nrows_list += [(num_rows, num_cols) for num_cols, num_rows in ncols_nrows_list[-1::-1]]

    # Compute best cols and rows

    ratio_dist_list = [
        abs(math.log(wh_ratio) - math.log((ncols * plt_w) / (nrows * plt_h)))
        for ncols, nrows in ncols_nrows_list
    ]

    curr_idx = 0
    while ratio_dist_list[curr_idx] > ratio_dist_list[curr_idx + 1] and curr_idx < len(ratio_dist_list) - 2:
        curr_idx += 1

    return ncols_nrows_list[curr_idx]
