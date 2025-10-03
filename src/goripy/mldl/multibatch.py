import numpy



def compute_multibatches_dataset(
    dataset_len_arr,
    dataloader_batch_size_arr,
    dataset_point_size_arr,
    min_step_items,
    world_size
):
    """
    Computes the number of batches per dataset and step in a multi-dataset, multi-gpu training.
    This version uses dataset lengths to keep track of dataloader progress.

    :param dataset_len_arr: numpy.ndarray
        1D numpy array with each dataset length.
    :param dataloader_batch_size_arr: numpy.ndarray
        1D numpy array with each dataloader batch size.
    :param dataset_point_size_arr: numpy.ndarray
        1D numpy array with the dataset point size (number of data points per element).
    :param min_step_items: int
        Minimum number of desired items in each step.
    :param world_size: int
        Number of gpus.

    :return: numpy.ndarray
        2D numpy array with the number of batches per step.
          - Axis 0: Step.
          - Axis 1: Dataset.
    """

    num_datasets = len(dataset_len_arr)
    
    dataset_curr_len_arr = numpy.zeros(shape=(num_datasets), dtype=numpy.uint32)

    step_num_batches_arr_list = []
    step_num_batches_arr = numpy.zeros(shape=(num_datasets), dtype=numpy.uint32)

    while not numpy.all(dataset_curr_len_arr == dataset_len_arr):

        step_num_batches_arr[:] = 0
        curr_step_items = 0

        while curr_step_items < min_step_items and not numpy.all(dataset_curr_len_arr == dataset_len_arr):

            dataset_prog_ratio_arr = dataset_curr_len_arr / dataset_len_arr
            dataset_idx = numpy.argmin(dataset_prog_ratio_arr)

            batch_points = min(
                dataloader_batch_size_arr[dataset_idx] * world_size,
                dataset_len_arr[dataset_idx] - dataset_curr_len_arr[dataset_idx]
            )

            batch_items = round(batch_points * dataset_point_size_arr[dataset_idx])

            step_num_batches_arr[dataset_idx] += 1
            dataset_curr_len_arr[dataset_idx] += batch_items

            curr_step_items += batch_items

        step_num_batches_arr_list.append(step_num_batches_arr.copy())

    return numpy.stack(step_num_batches_arr_list)



def compute_multibatches_dataloader(
    dataloader_len_arr,
    dataloader_batch_size_arr,
    dataset_point_size_arr,
    min_step_items,
    world_size
):
    """
    Computes the number of batches per dataset and step in a multi-dataset, multi-gpu training.
    This version uses dataloader lengths to keep track of dataloader progress.

    :param dataloader_len_arr: numpy.ndarray
        1D numpy array with each dataloader length.
    :param dataloader_batch_size_arr: numpy.ndarray
        1D numpy array with each dataloader batch size.
    :param dataset_point_size_arr: numpy.ndarray
        1D numpy array with the dataset point size (number of items per data point).
    :param min_step_items: int
        Minimum number of desired items in each step.
    :param world_size: int
        Number of gpus.

    :return: numpy.ndarray
        2D numpy array with the number of batches per step.
          - Axis 0: Step.
          - Axis 1: Dataset.
    """

    num_datasets = len(dataloader_len_arr)

    dataloader_curr_len_arr = numpy.zeros(shape=(num_datasets), dtype=numpy.uint32)
    dataloader_num_items_per_batch_arr = dataloader_batch_size_arr * dataset_point_size_arr * world_size

    step_num_batches_arr_list = []
    step_num_batches_arr = numpy.empty(shape=(num_datasets), dtype=numpy.uint32)

    while not numpy.all(dataloader_curr_len_arr == dataloader_len_arr):

        step_num_batches_arr[:] = 0
        curr_step_items = 0

        while curr_step_items < min_step_items and not numpy.all(dataloader_curr_len_arr == dataloader_len_arr):

            dataloader_prog_ratio_arr = dataloader_curr_len_arr / dataloader_len_arr
            dataloader_idx = numpy.argmin(dataloader_prog_ratio_arr)

            step_num_batches_arr[dataloader_idx] += 1
            curr_step_items += dataloader_num_items_per_batch_arr[dataloader_idx]

            dataloader_curr_len_arr[dataloader_idx] += 1

        step_num_batches_arr_list.append(step_num_batches_arr.copy())

    return numpy.stack(step_num_batches_arr_list)
