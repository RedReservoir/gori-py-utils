def add_newlines(orig_str, line_len=None):
    """
    Adds newlines to a string.

    :param orig_str: str
        Original one-line string.
    :param line_len: int, optional
        Line length.

    :return: str
        New multiline string.
    """

    start_stop_idxs = list(range(0, len(orig_str), line_len))
    if start_stop_idxs[-1] != len(orig_str): start_stop_idxs.append(len(orig_str))

    new_str = "\n".join(
        orig_str[start_idx:stop_idx]
        for start_idx, stop_idx in zip(start_stop_idxs[:-1], start_stop_idxs[1:])
    )

    return new_str


def add_newlines_whitespace(orig_str, min_len=None, max_len=None):
    """
    Adds newlines in place of whitespaces to a string.

    :param orig_str: str
        Original one-line string.
    :param min_len: int, optional
        Minimum line length.
    :param max_len: int, optional
        Maximum line length.

    :return: str
        New multiline string.
    """

    if min_len is None: min_len = float("inf")
    if max_len is None: max_len = float("inf")

    new_str = ""
    curr_line_len = 0

    for tkn in orig_str.split():

        if curr_line_len + len(tkn) > max_len:
            
            new_str = new_str[:-1] + "\n"
            new_str += tkn + " "
            curr_line_len = len(tkn) + 1

        else:

            new_str += tkn + " "
            curr_line_len += len(tkn) + 1

            if curr_line_len >= min_len:
                
                new_str = new_str[:-1] + "\n"
                curr_line_len = 0

    new_str = new_str[:-1]

    return new_str
