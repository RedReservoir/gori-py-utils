import torch

import goripy.memory.info



def sprint_tensor_info(
    tensor,
    name=None
):
    """
    Generates tensor information into a string.
    Used for debug purposes.

    Args:

        tensor (torch.Tensor):
            Tensor to print information from.

        name (str, optional):
            Name of the tensor variable. If not provided, name will not be shown.

    Returns:
    
        str:
            A string with the tensor information.
    """

    tensor_info_str = ""

    if name is not None:
        tensor_info_str += "name: {:s}".format(name)
        tensor_info_str += ", "
    tensor_info_str += "shape: " + str(tensor.shape)
    tensor_info_str += ", "
    tensor_info_str += "dtype: " + str(tensor.dtype)
    tensor_info_str += ", "
    tensor_info_str += "device: " + str(tensor.device)
    tensor_info_str += ", "
    tensor_info_str += "mem: " + goripy.memory.info.sprint_fancy_num_bytes(tensor.nelement() * tensor.element_size())

    return tensor_info_str



def sprint_tensor_stats(
    tensor,
    name=None
):
    """
    Generates tensor information into a string.
    Used for debug purposes.

    :param tensor: torch.Tensor
        Tensor to print information from.
    :param name: str, optional
        Name of the tensor variable.
        If not provided, name will not be shown.
    
    :return: str
        A string with the tensor information.
    """

    tensor_info_str = ""

    if name is not None:
        tensor_info_str += "name: {:s}".format(name)
        tensor_info_str += ", "
    
    with torch.no_grad():
        tensor_min = torch.min(tensor).item()
        tensor_max = torch.max(tensor).item()
        tensor_mean = torch.mean(tensor).item()
        tensor_std = torch.std(tensor).item()

    tensor_info_str += "min: " + str(tensor_min)
    tensor_info_str += ", "
    tensor_info_str += "max: " + str(tensor_max)
    tensor_info_str += ", "
    tensor_info_str += "mean: " + str(tensor_mean)
    tensor_info_str += ", "
    tensor_info_str += "std: " + str(tensor_std)

    return tensor_info_str
