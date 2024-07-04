from minio import Minio
def upload_files(files: list | str,endpoint,access_key=None,secrct_key=None):
    client=Minio(endpoint=endpoint,access_key=access_key,secret_key=secrct_key)