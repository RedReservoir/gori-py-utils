"""
Pickling utils.
"""



import pickle



def pickle_test(obj):
    """
    Checks if an object is pickable.
    Used for debug purposes.

    Args:
    
        obj (any):
            Object to pickle.

    Returns:

        str or bool: Returns True if object is pickable.
            Returns a string with the exception type and message otherwise.
    """
    
    try:
        pickle.dumps(obj)
    except Exception as ex:
        return "{:s}: {:s}".format(type(ex).__name__, str(ex))
    return True
