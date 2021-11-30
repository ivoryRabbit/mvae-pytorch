import boto3
import pandas as pd
from pandas import DataFrame

s3 = boto3.client("s3")
bucket_name = "data-bucket"


def get_key_from_s3(file_name: str) -> str:
    key = None

    s3_objects = s3.list_objects(Bucket=bucket_name, Prefix=file_name, Delimiter="_SUCCESS")
    for content in s3_objects["Contents"]:
        key = content["Key"]

    if key is None:
        raise
    return key


def read_csv_s3(file_name: str) -> DataFrame:
    key = get_key_from_s3(file_name)
    response = s3.get_object(Bucket=bucket_name, Key=key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        return pd.read_csv(response.get("Body"))
    else:
        raise


def read_parquet_s3(file_name: str) -> DataFrame:
    import io
    key = get_key_from_s3(file_name)
    response = s3.get_object(Bucket=bucket_name, Key=key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        return pd.read_parquet(io.BytesIO(response["Body"].read()))
    else:
        raise


def download_s3_object(file_name: str, save_path: str) -> None:
    s3.download_file(bucket_name, file_name, save_path)


def upload_parquet_to_s3(df: DataFrame, file_name, compression="snappy") -> None:
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression=compression, index=False)
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=buffer.getvalue())
