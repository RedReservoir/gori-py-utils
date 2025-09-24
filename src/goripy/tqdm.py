import tqdm



def tqdmidify(iter_obj, tqdm_len, tqdm_freq, tqdm_file):
    """
    Applies pre-formated tqdm wrapper on an iterable.

    :param iter_obj: Iterable
        Iterable object to wrap.
    :param tqdm_len: int
        Length of the iterable.
    :param tqdm_freq: int
        Number of tqdm progress bar updates.
    :param tqdm_file: any
        File object to use for printing output.

    :return: Iterable
        Iterable object wrapped with tqdm.
    """

    return tqdm.tqdm(
        iter_obj,
        total=tqdm_len,
        miniters=round(tqdm_len/tqdm_freq),
        ascii=False,
        desc="  Progress",
        ncols=0,
        mininterval=0,
        maxinterval=float("inf"),
        smoothing=0.99,
        file=tqdm_file
    )
