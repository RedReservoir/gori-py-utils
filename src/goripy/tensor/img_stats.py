import torch



def normalize_image_tensor(img):
    """
    Transforms a C x W x H float tensor with values rescaled into the [0, 1] interval
    (per channel).

    Args:

        img (torch.Tensor):
            Original image tensor.

    Returns:

        torch.Tensor:
            Normalized image tensor.
    """

    img_flt = img.flatten(start_dim=1, end_dim=2)

    img_max = img_flt.max(dim=1).values[:, None, None]
    img_min = img_flt.min(dim=1).values[:, None, None]

    return (img - img_min) / (img_max - img_min)



def standardize_image_tensor(img):
    """
    Transforms a C x W x H float tensor with values standardized (per channel).

    Args:

        img (torch.Tensor):
            Original image tensor.

    Returns:
    
        torch.Tensor:
            Standardized image tensor.
    """

    img_flt = img.flatten(start_dim=1, end_dim=2)

    img_mean = img_flt.mean(dim=1)[:, None, None]
    img_std = img_flt.std(dim=1)[:, None, None]

    return (img - img_mean) / img_std
