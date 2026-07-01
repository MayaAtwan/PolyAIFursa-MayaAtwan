"""Tiny S3 helper for the Agent service.

Bucket name and region are never hard-coded — they come from the environment:
  AWS_REGION     - e.g. "us-east-1"
  AWS_S3_BUCKET  - e.g. "maya-polyai-images"

Functions are kept small and named so tests can monkeypatch them without a real
S3 connection.
"""

import os

import boto3

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")

_s3 = None


def _client():
    """Create the boto3 S3 client lazily (so importing this module never fails)."""
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=AWS_REGION)
    return _s3


def upload_bytes(data: bytes, key: str, content_type: str = "image/jpeg") -> None:
    """Upload raw bytes to s3://AWS_S3_BUCKET/<key>."""
    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET environment variable is not set")
    _client().put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=data, ContentType=content_type)


def download_bytes(key: str) -> bytes:
    """Download s3://AWS_S3_BUCKET/<key> and return its bytes."""
    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET environment variable is not set")
    return _client().get_object(Bucket=AWS_S3_BUCKET, Key=key)["Body"].read()


def generate_presigned_url(key: str, expiry_seconds: int = 3600) -> str:
    """Return a pre-signed GET URL for s3://AWS_S3_BUCKET/<key>."""
    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET environment variable is not set")
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": AWS_S3_BUCKET, "Key": key},
        ExpiresIn=expiry_seconds,
    )
