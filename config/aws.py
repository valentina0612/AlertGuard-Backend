import boto3
from .settings import settings

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
    config=boto3.session.Config(
        connect_timeout=300,
        read_timeout=600,
        retries={'max_attempts': 3}
    )
)
