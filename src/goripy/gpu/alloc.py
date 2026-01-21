import os

import numpy

import goripy.file.json



class DeviceAllocator:
    """
    Manages GPU device allocation for multiple processes.
    Internal state is persisted by updating files form the local storage.
    Does not implement locking mechanisms.
    
    :param data_dirname: str
        Directory where to store internal state for this class. Must exist beforehand.
    :param device_cap_arr: numpy.ndarray
        A 1D signed integer numpy array with the device capacity values.
        Must contain values greater than zero.
    """

    def __init__(
        self,
        data_dirname,
        device_cap_arr
    ):

        self._data_dirname = data_dirname

        # Generate internal state and save

        self._device_cap_arr = device_cap_arr.copy()
        self._tenant_map = {}

        self._save()


    def allocate(
        self,
        tenant_id,
        req_device_cap
    ):
        """
        Allocates device capacity for a tenant, automatically selecting the device index.
        This method will fail if not enough device capacity is available.
        
        :param tenant_id: str
            ID of the tenant allocating device capacity.
        :param device_cap: int
            Amount of device capacity requested. Must be an integer greater than zero.
        
        :return: int
            The index of the granted device.
        """

        # Load internal state

        self._load()

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

        self._tenant_map[tenant_id] = {
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

        if tenant_id not in self._tenant_map:
            return

        device_idx = self._tenant_map[tenant_id]["device_idx"]
        req_device_cap = self._tenant_map[tenant_id]["req_device_cap"]

        # Update internal state and save

        self._device_cap_arr[device_idx] += req_device_cap

        del self._tenant_map[tenant_id]

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
        
        return list(self._tenant_map.keys())


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
            self._tenant_map,
            os.path.join(self._data_dirname, "tenant_map.json")
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

        self._tenant_map = goripy.file.json.load_json(
            os.path.join(self._data_dirname, "tenant_map.json")
        )
