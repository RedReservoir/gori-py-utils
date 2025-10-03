import os
import sys
import importlib



def load_module(
    module_path,
    dont_write_bytecode=True
):
    """
    Loads a Python module without saving it to sys.modules, so it can be deleted later.

    :param module_path: str
        Path to the Python module to load.
    :param sys_reg_module_name: str, optional
        Name to register the module under in `sys.modules`.
        If not provided, the module is not registered to `sys.modules`.
    :param dont_write_bytecode: bool, default=True
        If True, does not write `__pycache__` bytecode when loading the module.
    
    :return: any
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
