# servicios/mongo.py - VERSIÓN SÍNCRONA
from bson import ObjectId
from typing import Optional, Dict
from config.db import videos
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

def save_video_metadata(s3_key: str, url: str, size: int) -> str:
    """
    Guarda la metadata de un video en MongoDB (SÍNCRONO).
    """
    try:
        doc = {
            "_id": str(uuid.uuid4()),
            "s3_key": s3_key,
            "url": url,
            "size": size,
            "upload_date": datetime.utcnow(),
            "status": "active"
        }
        result = videos.insert_one(doc)  # ✅ SIN await
        logger.info(f"✅ Metadata guardada en MongoDB para: {s3_key}")
        return str(doc["_id"])
    except Exception as e:
        logger.error(f"❌ Error guardando en MongoDB: {str(e)}")
        raise

def get_video_metadata(s3_key: str) -> Optional[Dict]:
    try:
        return videos.find_one({"s3_key": s3_key})  # ✅ SIN await
    except Exception as e:
        logger.error(f"❌ Error obteniendo metadata: {str(e)}")
        return None

def delete_video_metadata(s3_key: str) -> bool:
    try:
        result = videos.delete_one({"s3_key": s3_key})  # ✅ SIN await
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Error eliminando metadata: {str(e)}")
        return False