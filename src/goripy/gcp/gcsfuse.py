import os



def gcs_uri_to_local_path(
    gcs_uri,
    gcsfuse_root
):
    """
    Converts a GCS URI to a local path in the context of gcsfuse.

    Args:

        gcs_uri (str):
            The GCS URI to convert.
        
        gcsfuse_root (str):
            Path to the mounted buckets with gcsfuse.
    
    Returns:

        str:
            The converted local path.
    """

    local_path = os.path.join(gcsfuse_root, gcs_uri[5:])
    return local_path



def local_path_to_gcs_uri(
    local_path,
    gcsfuse_root
):
    """
    Converts a local path to a GCS URI in the context of gcsfuse.

    Args:

        gcs_uri (str):
            The local path to convert.
        
        gcsfuse_root (str):
            Path to the mounted buckets with gcsfuse.
    
    Returns:

        str:
            The converted GCS URI.
    """

    gcs_uri = "gs://" + local_path.replace(gcsfuse_root, "")
    return gcs_uri
