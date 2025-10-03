import json



def save_json(
    obj,
    filename,
    indent=4
):
    """
    Saves an object to a JSON file.

    Args:
        obj (dict):
            Object to save.
        json_filename (str):
            Name of the JSON file.
        indent (int, optional):
            Number of spaces for indentation.
            Defaults to 4.
    """

    with open(filename, 'w') as json_file:
        json.dump(obj, json_file, indent=indent)



def load_json(
    filename
):
    """
    Loads an object from a JSON file.

    Args:
        filename (str):
            Name of the JSON file.

    Returns:
        any:
            Loaded object.
    """

    with open(filename, 'r') as json_file:
        obj = json.load(json_file)

    return obj
