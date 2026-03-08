import os
import pathlib



def parse_gcs_uri(
    gcs_uri
):
    """
    Parses a GCS URI.
    
    Args:

        gcs_uri (str):
            GCS URI to parse.
        
    Returns:

        2-tuple: A tuple containing:
            - `bucket_name`: Extracted bucket name.
            - `prefix`: Extracted prefix.
    """
    
    uri = gcs_uri.replace("gs://", "")
    bucket_name, prefix = uri.split("/", 1)
    return bucket_name, prefix



def download_gcs_file(
    gcs_client,
    bucket_name,
    src_filename,
    dst_filename
):
    """
    Downloads a file form a GCS bucket.
    
    Args:

        gcs_client (google.cloud.storage.client.Client):
            GCS client to use.
        bucket_name (str):
            Name of the bucket to download from.

        src_filename (str):
            Name of the file to download from the bucket.
        dst_filename (str):
            Name of the file to create and download into.
            An error will be raised if this file already exists.
    """

    if os.path.exists(dst_filename):
        raise ValueError("Destination file already exists")

    bucket = gcs_client.bucket(bucket_name)

    blob = bucket.blob(src_filename)
    blob.download_to_filename(dst_filename)



def download_gcs_directory(
    gcs_client,
    bucket_name,
    src_dirname,
    dst_dirname,
    depth=None
):
    """
    Downloads a directory form a GCS bucket.
    
    Args:

        gcs_client (google.cloud.storage.client.Client):
            GCS client to use.
        bucket_name (str):
            Name of the bucket to download from.

        src_dirname (str):
            Name of the directory to download from the bucket.
        dst_dirname (str):
            Name of the directory to create and download into.
            An error will be raised if this directory already exists.

        depth (int):
            Maximum recursive depth when downloading.
            If not provided, no limit will be applied.
    """

    if os.path.exists(dst_dirname):
        raise ValueError("Destination directory already exists")

    # Generate GCS blobs

    bucket = gcs_client.bucket(bucket_name)
    prefix = src_dirname

    for blob in gcs_client.list_blobs(bucket, prefix=prefix):

        blob_name = blob.name
        blob_rel_name = blob_name[len(prefix)+1:]
        
        # Skip if depth is exceeded

        blob_depth = 0 if blob_rel_name == "" else len(blob_rel_name.split(os.path.sep))
        if depth is not None and blob_depth > depth:
            continue

        # Create parent directories and download file

        blob_dst_path = pathlib.Path(dst_dirname) / blob_rel_name
        blob_dst_path_parent = pathlib.Path(blob_dst_path).parent

        os.makedirs(blob_dst_path_parent, exist_ok=True)

        blob_is_dir = blob_name.endswith(os.path.sep)
        if not blob_is_dir: blob.download_to_filename(blob_dst_path)