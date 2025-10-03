import os
import sys
import importlib



def load_module(
    module_path,
    dont_write_bytecode=True
):
    """
    Loads a Python module without saving it to sys.modules, so it can be deleted later.

    Args:
    
        module_path (str):
            Path to the Python module to load.

        dont_write_bytecode (bool, optional):
            If True, does not write `__pycache__` bytecode when loading the module.
            Defaults to True.

    Returns:

        any:
            The loaded Python module as an object.
    """

    module_name = os.path.splitext(os.path.basename(module_path))[0]

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(module_spec)

    if dont_write_bytecode:
        old_sys_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True

    module_spec.loader.exec_module(module)

    if dont_write_bytecode:
        sys.dont_write_bytecode = old_sys_dont_write_bytecode

    return module
