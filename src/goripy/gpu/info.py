import torch

import goripy.memory.info



def sprint_device_memory_usage(
    device
):
    """
    Prints torch device information into a string.

    Args:
    
        device (torch.device):
            Device to print information from.

    Returns:

        str:
            A string with the device information.
    """

    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(device)
    used_mem_bytes = total_mem_bytes - free_mem_bytes
    
    used_mem_str = goripy.memory.info.sprint_fancy_num_bytes(used_mem_bytes, unit="GiB")
    total_mem_str = goripy.memory.info.sprint_fancy_num_bytes(total_mem_bytes, unit="GiB")

    device_msg = "Allocated memory: [{:s}] / [{:s}]".format(
        used_mem_str, total_mem_str
    )

    return device_msg
