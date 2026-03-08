import math
import numpy



def compute_batch_instance_index_limits(
    num_instances,
    max_batch_size
):
    """
    Generates batch limits (starting and ending indices).

    Args:

        num_instances (int):
            Number of instances.

        max_batch_size (int):
            Maximum allowed batch size.

    Returns:

        2-tuple of numpy.ndarray:
            Two 1D numpy arrays with each batch limits.
    """

    batch_instance_idx_limit_arr = numpy.arange(math.ceil(num_instances / max_batch_size) + 1)
    batch_instance_idx_limit_arr *= max_batch_size
    batch_instance_idx_limit_arr[-1] = num_instances

    return (
        batch_instance_idx_limit_arr[:-1].copy(),
        batch_instance_idx_limit_arr[1:].copy()
    )
