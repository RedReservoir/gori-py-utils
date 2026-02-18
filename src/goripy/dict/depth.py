import goripy.uuid



def flatten(
    nest_dict
):
    """
    Flattens a dict so that it is only one level deep.

    Args:

        nest_dict (dict):
            Nested dict to flatten.

    Returns:

        tuple:
            A 2-tuple consisting of:

              - `flat_dict`: The flattened dict.
              - `key_map`: The map from nested to flattened dict keys.
    """

    flat_dict = {}

    key_map =_flatten_recursive(
        nest_dict,
        flat_dict
    )

    return flat_dict, key_map
    


def _flatten_recursive(
    nest_dict,
    flat_dict
):

    key_map = {}

    for key, value in nest_dict.items():
        
        if type(value) is dict:
            sub_key_map = _flatten_recursive(value, flat_dict)
            key_map[key] = sub_key_map
        else:
            key_uuid = goripy.uuid.get_new_uuid(uuid_set=flat_dict, update_uuid_set=False)
            key_map[key] = key_uuid
            flat_dict[key_uuid] = value
    
    return key_map



def unflatten(
    flat_dict,
    key_map
):
    """
    Inverse of the `flatten` method.

    Args:

        flat_dict (dict):
            The flattened dict produced by `flatten`.

        key_map (dict):
            The key map produced by `flatten`.

    Returns:

        dict:
            The original nested dict.
    """

    nest_dict = _unflatten_recursive(
        flat_dict,
        key_map
    )

    return nest_dict



def _unflatten_recursive(
    flat_dict,
    key_map
):
    
    nest_dict = {}
    
    for key, value in key_map.items():

        if type(value) is dict:
            nest_dict[key] = _unflatten_recursive(flat_dict, value)
        else:
            nest_dict[key] = flat_dict[value]

    return nest_dict
