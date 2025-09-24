import json



def save_json(
        obj,
        filename,
        indent=4
        ):
    """
    Saves an object to a JSON file.

    :param obj: dict
        Object to save.
    :param json_filename: str
        Name of the JSON file.
    :param indent: int, default=4
        Number of spaces for indentation.
    """

    with open(filename, 'w') as json_file:
        json.dump(obj, json_file, indent=indent)



def load_json(
        filename
        ):
    """
    Loads an object from a JSON file.

    :param filename: str
        Name of the JSON file.

    :return: any
        Loaded object.
    """

    with open(filename, 'r') as json_file:
        obj = json.load(json_file)

    return obj
