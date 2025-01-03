import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def download_s3_bucket(bucket_name, local_dir):
    """
    Downloads all objects from an S3 bucket to a local directory.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        local_dir (str): Local directory to save the downloaded files.

    Returns:
        None
    """
    s3 = boto3.client('s3')

    try:
        os.makedirs(local_dir, exist_ok=True)

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Get the S3 key (path) of the object
                    s3_key = obj['Key']

                    # Skip if it's a folder (S3 keys ending with '/')
                    if s3_key.endswith('/'):
                        continue

                    # Construct the local file path
                    local_file_path = os.path.join(local_dir, s3_key)

                    # Ensure the local directory structure exists
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    # Download the file
                    s3.download_file(bucket_name, s3_key, local_file_path)
                    print(f"Downloaded: s3://{bucket_name}/{s3_key} to {local_file_path}")
            else:
                print(f"No objects found in s3://{bucket_name}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials.")
    except Exception as e:
        print(f"An error occurred: {e}")

s3=boto3.client('s3')
bucket_name='tinybertsentimentanalysis'


def upload_image_to_s3(file_name, s3_prefix="ml-images", object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    object_name = f"{s3_prefix}/{object_name}"

    s3.upload_file(file_name, bucket_name, object_name)

    response = s3.generate_presigned_url('get_object',
                                         Params={
                                             "Bucket": bucket_name,
                                             "Key": object_name
                                         },
                                         ExpiresIn=3600)
    
    return response
