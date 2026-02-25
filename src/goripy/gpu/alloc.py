import os
import shutil

import numpy

import goripy.file.json



class DeviceAllocator:
    """
    Manages GPU device allocation for multiple processes.
    Internal state is persisted by updating local storage files.
    Calls to methods of this class should be surrounded by locks.
    
    :param data_dirname: str
        Directory where to store internal state.
    """

    def __init__(
        self,
        data_dirname
    ):

        self._data_dirname = data_dirname


    @staticmethod
    def initialize_data(
        data_dirname,
        device_cap_arr
    ):
        """
        Initializes fresh data to be read by instances of this class.

        :param data_dirname: str
            Directory where to store internal state.
        :param device_cap_arr: numpy.ndarray
            A 1D unsigned integer numpy array with the device capacity values.
            Must contain values greater than zero.
        """
        
        device_alloc_signal_filename = os.path.join(data_dirname, "device_alloc.txt")

        # Delete old internal state data directory
        # Implements safeguard not to accidentally delete important directories

        if os.path.exists(data_dirname):
            
            if not os.path.exists(device_alloc_signal_filename):

                err_msg = "Directory \"{:s}\" does not contain device allocator data".format(
                    data_dirname
                )

                raise ValueError(err_msg)

            shutil.rmtree(data_dirname)

        # Create new internal state data directory

        os.mkdir(data_dirname)

        with open(device_alloc_signal_filename, "w"):
            pass

        numpy.save(
            os.path.join(data_dirname, "device_cap_arr.npy"),
            device_cap_arr
        )

        goripy.file.json.save_json(
            {},
            os.path.join(data_dirname, "tenant_alloc_map.json")
        )


    def allocate(
        self,
        tenant_id,
        req_device_cap
    ):
        """
        Allocates device capacity for a tenant, automatically selecting the device index.
        This method will fail if not enough device capacity is available.
        
        Prioritizes devices with the least remaining capacity.

        :param tenant_id: str
            ID of the tenant allocating device capacity.
        :param device_cap: int
            Amount of device capacity requested. Must be an integer greater than zero.
        
        :return: int
            The index of the granted device.
        """

        # Load internal state

        self._load()

        # If tenant ID already exists, error

        if tenant_id in self._tenant_alloc_map:

            err_msg = "Tenant ID ({:s}) already exists".format(
                tenant_id,
            )
            
            raise ValueError(err_msg)

        # Select available resource index to allocate

        av_device_idx_arr = numpy.flatnonzero(self._device_cap_arr >= req_device_cap)

        if av_device_idx_arr.size == 0:
            
            err_msg = "Requested capacity ({:d}) exceeds available capacity ({:d})".format(
                req_device_cap,
                numpy.max(self._device_cap_arr)
            )
            
            raise ValueError(err_msg)
        
        av_device_zidx = numpy.argmin(self._device_cap_arr[av_device_idx_arr])
        device_idx = av_device_idx_arr[av_device_zidx]

        # Update internal state and save

        self._device_cap_arr[device_idx] -= req_device_cap

        self._tenant_alloc_map[tenant_id] = {
            "device_idx": int(device_idx),
            "req_device_cap": int(req_device_cap)
        }

        self._save()

        return device_idx
    

    def deallocate(
        self,
        tenant_id
    ):
        """
        Deallocates device capacity allocated by a tenant.
        This method will do nothing if the provided tenant ID does not exist.

        :param tenant_id: str
            ID of the tenant with allocated device capacity.
        """

        # Load internal state

        self._load()

        # Load tenant allocation data

        if tenant_id not in self._tenant_alloc_map:
            return

        device_idx = self._tenant_alloc_map[tenant_id]["device_idx"]
        req_device_cap = self._tenant_alloc_map[tenant_id]["req_device_cap"]

        # Update internal state and save

        self._device_cap_arr[device_idx] += req_device_cap

        del self._tenant_alloc_map[tenant_id]

        self._save()


    def get_tenant_ids(
        self
    ):
        """
        Returns list with all currently existing tenant IDs.

        :return: list of str
            A list with all currently existing tenant IDs.
        """

        # Load internal state

        self._load()
        
        return list(self._tenant_alloc_map.keys())


    def _save(
        self
    ):
        """
        Saves internal status to local storage. 
        """

        numpy.save(
            os.path.join(self._data_dirname, "device_cap_arr.npy"),
            self._device_cap_arr
        )

        goripy.file.json.save_json(
            self._tenant_alloc_map,
            os.path.join(self._data_dirname, "tenant_alloc_map.json")
        )
        

    def _load(
        self
    ):
        """
        Reads internal status from local storage. 
        """

        self._device_cap_arr = numpy.load(
            os.path.join(self._data_dirname, "device_cap_arr.npy")
        )

        self._tenant_alloc_map = goripy.file.json.load_json(
            os.path.join(self._data_dirname, "tenant_alloc_map.json")
        )
