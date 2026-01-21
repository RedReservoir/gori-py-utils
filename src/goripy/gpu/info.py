import torch

import goripy.memory.info



def sprint_device_info(
    device
):
    """
    Generates torch device information into a string.
    Used for debug purposes.

    Args:
    
        device (torch.device):
            Device to print information from.

    Returns:

        str:
            A string with the device information.
    """

    device_id = torch.cuda._get_nvml_device_index(device)
    device_name = torch.cuda.get_device_name(device)

    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(device)
    used_mem_bytes = total_mem_bytes - free_mem_bytes
    
    used_mem_str = goripy.memory.info.sprint_fancy_num_bytes(used_mem_bytes, unit="GiB")
    total_mem_str = goripy.memory.info.sprint_fancy_num_bytes(total_mem_bytes, unit="GiB")

    gpu_log_msg = "Device ID {:d} ({:s}) - Allocated memory: [{:s}] / [{:s}]".format(
        device_id, device_name, used_mem_str, total_mem_str
    )

    return gpu_log_msg
