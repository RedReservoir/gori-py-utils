import os

import numpy

import goripy.memory.get



class VariableLength2DListStorage:
    """
    Stores a 2D list of variable length into numpy arrays for more efficient memory usage.
    Each 1D list is returned as a numpy array.
    """


    def __init__(
        self
    ):

        self._initialized = False


    def __getitem__(
        self,
        idx
    ):

        if not self._initialized:
            raise ValueError("Data has not been initialized")

        return self._value_arrr[idx, :self._value_len_arr[idx]]


    def fill_2d_list(
        self,
        orig_value_llist,
        value_numpy_dtype,
        len_numpy_dtype
    ):
        """
        Fills this object with data coming from a 2D list.

        Args:
        
            orig_value_llist (list):
                2D list with the values to store.

            value_numpy_dtype (any):
                Numpy data type to use for value storage.

            len_numpy_dtype (any):
                Numpy data type to use for value length storage.
        """

        value_len_arr = numpy.fromiter((len(value_list) for value_list in orig_value_llist), dtype=len_numpy_dtype)

        value_arrr = numpy.empty(shape=(len(orig_value_llist), numpy.max(value_len_arr)), dtype=value_numpy_dtype)
        for idx, value_list in enumerate(orig_value_llist): value_arrr[idx, :value_len_arr[idx]] = value_list

        self._value_arrr = value_arrr
        self._value_len_arr = value_len_arr

        self._initialized = True


    def fill_2d_array(
        self,
        orig_value_arrr,
        value_invalid,
        value_numpy_dtype,
        len_numpy_dtype
    ):
        """
        Fills this object with data coming from a 2D array.
        Expects "empty" positions filled with an "invalid" value.

        Args:

            orig_value_arrr (numpy.ndarray):
                2D array with the values to store.

            value_invalid (any):
                Value used to fill "empty" positions in `orig_value_arr`.

            value_numpy_dtype (any):
                Numpy data type to use for value storage.

            len_numpy_dtype (any):
                Numpy data type to use for value length storage.
        """

        value_len_arr = numpy.sum(orig_value_arrr != value_invalid, axis=1).astype(len_numpy_dtype)
        value_arrr = orig_value_arrr.astype(value_numpy_dtype)[:, :numpy.max(value_len_arr)]

        self._value_arrr = value_arrr
        self._value_len_arr = value_len_arr

        self._initialized = True


    def save(
        self,
        filename
    ):
        """
        Stores data from this object into a file.
        Data is saved into an .npz file.

        Args:

            filename (str):
                Filename to save the data to.
        """

        if not self._initialized:
            raise ValueError("Data has not been initialized")

        numpy.savez(
            filename,
            value_arrr=self._value_arrr,
            value_len_arr=self._value_len_arr
        )

    
    def load(
        self,
        filename
    ):
        """
        Loads data to this object from a file.
        Data is loaded from an .npz file.

        Args:

            filename (str):
                Filename where the data is saved to.
        """

        numpy_data = numpy.load(filename)

        self._value_arrr = numpy_data["value_arrr"]
        self._value_len_arr = numpy_data["value_len_arr"]

        self._initialized = True


    def get_num_bytes(
        self
    ):
        """
        Computes the number of bytes that this object weights.

        Returns:

            int:
                Number of bytes that the object weights.
        """

        if not self._initialized:
            raise ValueError("Data has not been initialized")

        #

        num_bytes = 0

        num_bytes += goripy.memory.get.get_obj_bytes(self._initialized)
        num_bytes += goripy.memory.get.get_obj_bytes(self._value_arrr)
        num_bytes += goripy.memory.get.get_obj_bytes(self._value_len_arr)

        return num_bytes



def save_storage_dict(
    storage_dict,
    dirname
):
    """
    Saves a (possibly nested) dict where all leaf elements are VariableLength2DListStorage objects
    into a directory.

    Args:

        storage_dict (dict):
            The dict with storage objects.

        dirname (str):
            Name of the directory to save into.

    Returns:

        dict:
            A (possibly nested) dict with all saved storage objects.
    """
    
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    for key, value in storage_dict.items():

        if type(value) is dict:            

            dict_subdirname = os.path.join(dirname, key)
            save_storage_dict(value, dict_subdirname)
        
        elif type(value) is VariableLength2DListStorage:

            storage_filename = os.path.join(dirname, key + ".npz")
            value.save(storage_filename)
        
        else:

            raise ValueError("Invalid value type found. Expected {:s} or {:s}, found {:s}".format(
                str(dict),
                str(VariableLength2DListStorage),
                str(type(value))
            ))



def load_storage_dict(
    dirname
):
    """
    Loads a (possibly nested) dict where all leaf elements are VariableLength2DListStorage objects
    from a directory.

    Args:

        dirname (str):
            Name of the directory to save into.

    Returns:

        dict:
            The dict with storage objects.
    """

    storage_dict = {}

    for subname in os.listdir(dirname):

        full_subname = os.path.join(dirname, subname)

        if os.path.isfile(full_subname):
            storage = VariableLength2DListStorage()
            storage.load(full_subname)
            storage_dict[subname.split(".")[0]] = storage

        if os.path.isdir(full_subname):
            storage_dict[subname] = load_storage_dict(full_subname)

    return storage_dict
