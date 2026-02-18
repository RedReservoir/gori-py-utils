import os

import numpy

import goripy.memory.get



########



class VariableLength2DListStorage:
    """
    Stores and facilitates access of multiple variable length data.
    Indexing this object returns a 1D numpy array if variable length.
    
    Args:
    
        value_arrr (list):
            2D numpy array with the stored values.

        value_len_arr (any):
            1D numpy array with the length of each row. 
    """


    def __init__(
        self,
        value_arrr,
        value_len_arr
    ):
        
        self._value_arrr = value_arrr
        self._value_len_arr = value_len_arr


    def __getitem__(
        self,
        idx
    ):

        return self._value_arrr[idx, :self._value_len_arr[idx]]


    @classmethod
    def from_2d_list(
        cls,
        orig_value_llist,
        value_numpy_dtype,
        len_numpy_dtype
    ):
        """
        Creates a VariableLength2DListStorage from data coming from a 2D list.

        Args:
        
            orig_value_llist (list):
                2D list with the values to store.

            value_numpy_dtype (any):
                Numpy data type to use for value storage.

            len_numpy_dtype (any):
                Numpy data type to use for value length storage.

        Return:

            VariableLength2DListStorage:
                The created storage object.
        """

        value_len_arr = numpy.fromiter((len(value_list) for value_list in orig_value_llist), dtype=len_numpy_dtype)

        value_arrr = numpy.empty(shape=(len(orig_value_llist), numpy.max(value_len_arr)), dtype=value_numpy_dtype)
        for idx, value_list in enumerate(orig_value_llist): value_arrr[idx, :value_len_arr[idx]] = value_list

        return cls(value_arrr, value_len_arr)


    @classmethod
    def from_2d_numpy_array(
        cls,
        orig_value_arrr,
        value_invalid,
        value_numpy_dtype,
        len_numpy_dtype
    ):
        """
        Creates a VariableLength2DListStorage from data coming from a 2D numpy array.
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

        Return:

            VariableLength2DListStorage:
                The created storage object.
        """

        value_len_arr = numpy.sum(orig_value_arrr != value_invalid, axis=1).astype(len_numpy_dtype)
        value_arrr = orig_value_arrr.astype(value_numpy_dtype)[:, :numpy.max(value_len_arr)]

        return cls(value_arrr, value_len_arr)


    def save(
        self,
        filename
    ):
        """
        Saves this VariableLength2DListStorage into an `.npz` file.

        Args:

            filename (str):
                Filename to save to.
        """

        numpy.savez(
            filename,
            value_arrr=self._value_arrr,
            value_len_arr=self._value_len_arr
        )


    @classmethod
    def load(
        cls,
        filename
    ):
        """
        Loads a VariableLength2DListStorage from data coming from an `.npz` file.

        Args:

            filename (str):
                Filename to load from.

        Return:

            VariableLength2DListStorage:
                The loaded storage object.
        """

        numpy_data = numpy.load(filename)

        value_arrr = numpy_data["value_arrr"]
        value_len_arr = numpy_data["value_len_arr"]

        return cls(value_arrr, value_len_arr)


    def get_num_bytes(
        self
    ):
        """
        Computes the RAM memory overhead of this object.

        Returns:

            int:
                Number of bytes occupied by this object.
        """

        num_bytes = 0

        num_bytes += self._value_arrr.nbytes
        num_bytes += self._value_len_arr.nbytes

        return num_bytes



########



def save_storage_dict(
    storage_dict,
    dirname
):
    """
    Saves a dict where all leaf elements are VariableLength2DListStorage or `None` objects.
    Data is saved into a directory reproducing the dict structure.

    Args:

        storage_dict (dict):
            The dict containing the storage objects.

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

            storage_filename = os.path.join(dirname, "{:s}.npz".format(key))
            value.save(storage_filename)
        
        elif value is None:

            storage_filename = os.path.join(dirname, "{:s}.npz".format(key))
            numpy.savez(storage_filename, inv=numpy.asarray([]))

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
    Loads a dict where all leaf elements are VariableLength2DListStorage or `None` objects.
    Data is loaded from a directory reproducing the dict structure.

    Args:

        dirname (str):
            Name of the directory to save into.

    Returns:

        dict:
            A dict containing the storage objects.
    """

    storage_dict = {}

    for subname in os.listdir(dirname):

        full_subname = os.path.join(dirname, subname)

        if os.path.isfile(full_subname):

            npz_data = numpy.load(full_subname)

            if "inv" in npz_data:
                storage = None
            else:
                storage = VariableLength2DListStorage.load(full_subname)
            
            storage_dict[subname.split(".")[0]] = storage

        if os.path.isdir(full_subname):
            storage_dict[subname] = load_storage_dict(full_subname)

    return storage_dict
