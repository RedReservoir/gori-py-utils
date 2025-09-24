import logging



class TqdmLogger:
    """
    Writing class meant to be passed to tqdm when logging to a file.

    Args:

        logger (logging.Logger):
            Logger used for logging.

        log_level (int, optional):
            Log level to use when logging.
            Defaults to logging.INFO.
    """
    
    def __init__(
        self,
        logger,
        log_level=logging.INFO
    ):

        self._logger = logger
        self._log_level = log_level


    def write(self, message):

        message = message.replace("\n", "")
        message = message.replace("\r", "")
        
        if message != "":
            self._logger.log(self._log_level, message)


    def flush(self):
        pass



class StderrLogger:
    """
    Writing class meant to substitute sys.stderr for redirection.

    Args:

        logger (logging.Logger):
            Logger used for logging.
            
        log_level (int, optional):
            Log level to use when logging.
            Defaults to logging.ERROR.
    """
    
    def __init__(
        self,
        logger,
        log_level=logging.ERROR
    ):

        self._logger = logger
        self._log_level = log_level

        self._log_buff = ""


    def __del__(self):

        if self._log_buff != "":
            self._logger.log(self._log_level, self._log_buff)
            self._log_buff = ""


    def write(self, message):

        message_chunk_list = message.split("\n")

        self._log_buff += message_chunk_list[0]
        for message_chunk in message_chunk_list[1:]:

            if self._log_buff != "":
                self._logger.log(self._log_level, self._log_buff)
                self._log_buff = ""
            
            self._log_buff += message_chunk


    def flush(self):
        pass
