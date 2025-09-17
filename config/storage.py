import aioboto3
from botocore.config import Config
from config.settings import settings
from typing import Tuple, Optional, AsyncGenerator
import logging
from fastapi import HTTPException, status
from .settings import settings
logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.session = aioboto3.Session()
        self.s3_config = Config(
            connect_timeout=300,
            read_timeout=600,
            retries={'max_attempts': 3}
        )
        self.aws_region = "us-east-2"  # Ohio region
        self.default_content_type = "video/mp4"
        self.chunk_size = 1024 * 1024  # 1MB chunk size for streaming

    async def check_video_exists(self, s3_key: str) -> bool:
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.aws_region,
                config=self.s3_config
            ) as s3:
                response = await s3.head_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
                return response['ContentLength'] > 0
        except Exception as e:
            logger.error(f"Error checking video in S3: {str(e)}", exc_info=True)
            return False

    async def get_video_size(self, s3_key: str) -> int:
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.aws_region,
                config=self.s3_config
            ) as s3:
                response = await s3.head_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
                return response['ContentLength']
        except Exception as e:
            logger.error(f"Error getting video size: {str(e)}", exc_info=True)
            return 0

    async def upload_to_s3(self, file_obj, s3_key: str, content_type: str = None) -> Tuple[str, str]:
        try:
            final_content_type = content_type if content_type else self.default_content_type

            if hasattr(file_obj, 'seek'):
                file_obj.seek(0, 2)
                file_size = file_obj.tell()
                file_obj.seek(0)
            else:
                file_size = len(file_obj.getvalue()) if hasattr(file_obj, 'getvalue') else 0

            if file_size == 0:
                raise ValueError("File is empty")

            async with self.session.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.aws_region,
                config=self.s3_config
            ) as s3:
                await s3.upload_fileobj(
                    file_obj,
                    settings.S3_BUCKET_NAME,
                    s3_key,
                    ExtraArgs={
                        'ContentType': final_content_type,
                        'Metadata': {'original-size': str(file_size)}
                    }
                )

            uploaded_size = await self.get_video_size(s3_key)
            if uploaded_size != file_size:
                await self.delete_video(s3_key)
                raise ValueError(f"Size mismatch (uploaded: {uploaded_size}, original: {file_size})")

            url = f"https://{settings.S3_BUCKET_NAME}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            logger.info(f"File uploaded successfully to S3: {s3_key}, size: {uploaded_size} bytes")
            return s3_key, url

        except Exception as e:
            logger.error(f"S3 upload error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to upload file: {str(e)}"
            )

    async def get_video_stream(self, s3_key: str, chunk_size: Optional[int] = None) -> AsyncGenerator[bytes, None]:
        try:
            if not await self.check_video_exists(s3_key):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found in storage")

            video_size = await self.get_video_size(s3_key)
            if video_size == 0:
                logger.error(f"Video {s3_key} exists but is empty (0 bytes)")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Empty video file")

            async with self.session.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.aws_region,
                config=self.s3_config
            ) as s3:
                response = await s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)

                stream = response.get('Body', None)
                if stream is None:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No stream available")

                current_chunk_size = chunk_size if chunk_size else self.chunk_size

                try:
                    while True:
                        chunk = await stream.read(current_chunk_size)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    # close stream only if it's not None and supports async close
                    close_method = getattr(stream, 'close', None)
                    if close_method:
                        # aiobotocore stream.close() is async
                        if callable(getattr(close_method, '__await__', None)):
                            await stream.close()
                        else:
                            stream.close()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"S3 stream error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to stream video: {str(e)}"
            )

    async def delete_video(self, s3_key: str) -> bool:
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.aws_region,
                config=self.s3_config
            ) as s3:
                await s3.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
            logger.info(f"Video deleted from S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting video: {str(e)}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete video: {str(e)}")
        
        from .mongo_service import save_video_metadata, get_video_metadata, delete_video_metada