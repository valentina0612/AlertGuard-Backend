# servicios/mongo.py - VERSIÓN SÍNCRONA
from typing import Optional, Dict
from config.db import videos
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

def save_video_metadata(s3_key: str, url: str, size: int, results: Optional[Dict] = None) -> str:
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
        if results:
            doc["results"] = results  #agrega resultados si los pasas

        videos.insert_one(doc)
        logger.info(f"✅ Metadata guardada en MongoDB para: {s3_key}")
        return str(doc["_id"])
    except Exception as e:
        logger.error(f"❌ Error guardando en MongoDB: {str(e)}")
        raise


def update_video_results(video_id: str, results: Dict) -> bool:
    """
    Actualiza los resultados de análisis asociados a un video.
    """
    try:
        result = videos.update_one(
            {"_id": video_id},  # 👈 video_id ya es string (UUID)
            {"$set": {"results": results, "last_updated": datetime.utcnow()}}
        )
        if result.modified_count > 0:
            logger.info(f"✅ Resultados actualizados para video {video_id}")
            return True
        else:
            logger.warning(f"⚠️ No se actualizó nada. ¿Existe el video_id {video_id}?")
            return False
    except Exception as e:
        logger.error(f"❌ Error actualizando resultados: {str(e)}")
        return False


def get_video_metadata(s3_key: str) -> Optional[Dict]:
    try:
        return videos.find_one({"s3_key": s3_key})  # ✅ SIN await
    except Exception as e:
        logger.error(f"❌ Error obteniendo metadata: {str(e)}")
        return None

def get_video_with_results(video_id: str) -> Optional[Dict]:
    """
    Devuelve metadata + resultados de un video.
    """
    try:
        return videos.find_one({"_id": video_id})
    except Exception as e:
        logger.error(f"❌ Error obteniendo video: {str(e)}")
        return None


def delete_video_metadata(s3_key: str) -> bool:
    try:
        result = videos.delete_one({"s3_key": s3_key})  # ✅ SIN await
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Error eliminando metadata: {str(e)}")
        return False
