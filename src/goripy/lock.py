"""
Process locking utils.
"""



import fcntl



class FileLock:
    """
    Implements a file-based lock.

    :param lock_filename: str
        Name of the file to use as lock.
    """

    def __init__(
        self,
        lock_filename
    ):
    
        self.lock_filename = lock_filename
        self.lock_file = None


    def acquire(self):
        """
        Acquires the file-based lock.
        """

        self.lock_file = open(self.lock_filename, "w")
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)


    def release(self):
        """
        Releases the file-based lock.
        """
        
        if self.lock_file is not None:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None


    def __enter__(self):
        return self.acquire()


    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
