from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    MONGODB_URI: str
    MONGODB_DB_NAME: str = "AlertGuard-videos" 
    MONGO_VIDEO_COLLECTION: str = "videos"  # ← AÑADE ESTA LÍNEA
    S3_BUCKET_NAME: str
    AWS_REGION: Optional[str] = "us-east-2"

    class Config:
        env_file = ".env"

settings = Settings()