from bson import ObjectId
from typing import Optional, Dict
from config.db import videos  # tu instancia de la colección
from config.settings import settings

async def save_video_metadata(s3_key: str, url: str, size: int) -> str:
    doc = {
        "s3_key": s3_key,
        "url": url,
        "size": size,
    }
    result = videos.insert_one(doc)
    return str(result.inserted_id)

async def get_video_metadata(s3_key: str) -> Optional[Dict]:
    return videos.find_one({"s3_key": s3_key})

async def delete_video_metadata(s3_key: str) -> bool:
    result = videos.delete_one({"s3_key": s3_key})
    return result.deleted_count > 0
