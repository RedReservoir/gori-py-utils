import torch

import goripy.memory.info



def sprint_device_info(
    device
):
    """
    TODO
    """

    device_id = torch.cuda._get_nvml_device_index(device)
    device_name = torch.cuda.get_device_name(device)

    free_mem, total_mem = torch.cuda.mem_get_info(device)

    gpu_log_msg = "Device ID {:d} - {:s} - Allocated {:s} / {:s}".format(
        device_id,
        device_name,
        goripy.memory.info.sprint_fancy_num_bytes(total_mem - free_mem),
        goripy.memory.info.sprint_fancy_num_bytes(total_mem)
    )

    return gpu_log_msg
