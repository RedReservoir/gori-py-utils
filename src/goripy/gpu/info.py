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



def sprint_device_memory_usage_detail(
    device
):
    """
    Prints torch device information into a string.
    More detailed version.

    Args:
    
        device (torch.device):
            Device to print information from.

    Returns:

        str:
            A string with the device information.
    """

    mem_alloc_bytes = torch.cuda.memory_allocated(device)
    max_mem_alloc_bytes = torch.cuda.max_memory_allocated(device)
    mem_res_bytes = torch.cuda.memory_reserved(device)
    _, gpu_mem_bytes = torch.cuda.mem_get_info(device)

    device_msg = ""
    device_msg += "Curr. Allocated: {:s}".format(
        goripy.memory.info.sprint_fancy_num_bytes(mem_alloc_bytes, unit="GiB")
    )
    device_msg += ", "
    device_msg += "Max. Allocated: {:s}".format(
        goripy.memory.info.sprint_fancy_num_bytes(max_mem_alloc_bytes, unit="GiB")
    )
    device_msg += ", "
    device_msg += "Reserved: {:s}".format(
        goripy.memory.info.sprint_fancy_num_bytes(mem_res_bytes, unit="GiB")
    )
    device_msg += ", "
    device_msg += "Capacity: {:s}".format(
        goripy.memory.info.sprint_fancy_num_bytes(gpu_mem_bytes, unit="GiB")
    )

    return device_msg
