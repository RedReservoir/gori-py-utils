def sprint_fancy_num_bytes(num_bytes, unit=None):
    """
    Generates a fancy string representation of memory quantity.

    :param num_bytes: float
        Number of bytes.
    :param unit: str
        Desired unit


    :return: str
        Fancy string representation of the memory quantity.
    """
    
    units = ["B", "KiB", "MiB", "GiB"]
    unit_idx = 0

    while num_bytes > 1024 or ((unit is not None) and (unit != units[unit_idx])):
        num_bytes /= 1024
        unit_idx += 1

    byte_fancy_str = "{:.2f} {:3s}".format(num_bytes, units[unit_idx])

    return byte_fancy_str
