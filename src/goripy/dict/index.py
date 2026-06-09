def vals_to_idx_dict(vals):
    """
    Generates a value-to-index mapping using a dict.

    Args:

        vals (iterable of any):
            Iterable of values to map.

    Returns:

        dict:
            The generated value-to-index map.
    """

    return {val: idx for idx, val in enumerate(vals)}



def invert(my_dict):
    """
    Inverts a dict, turning keys and values into values and keys.

    Args:

        my_dict (dict):
            Dict to invert.

    Returns:

        dict:
            The inverted dict.
    """
    return {val: key for key, val in my_dict.items()}
