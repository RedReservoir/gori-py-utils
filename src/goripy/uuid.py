import uuid



def get_new_uuid(
    uuid_set=None,
    update_uuid_set=False
):
    """
    Generates a new UUID.

    :param uuid_set: set, optional
        UUID set to check for already existing UUIDs.
        No checks are made when no UUID set is passed.
    :param update_uuid_set: bool, default=False
        Whether to update the UUID set with the new UUID.
        Ignored when no UUID set is passed.
    
    :return str:
        The new generated UUID.
    """

    new_uuid = None
    while new_uuid is None:
        new_uuid = str(uuid.uuid4()).replace("-", "")
        if (uuid_set is not None) and (new_uuid in uuid_set): new_uuid = None
    if (uuid_set is not None) and update_uuid_set: uuid_set.add(new_uuid)

    return new_uuid
