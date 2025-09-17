from config.storage import StorageService
from servicios.mongo import save_video_metadata, get_video_metadata, delete_video_metadata
from .settings import settings
class VideoManager:
    def __init__(self, storage_service: StorageService):
        self.storage = storage_service

    async def upload_video(self, file, s3_key: str):
        # Subir a S3
        s3_key, url = await self.storage.upload_to_s3(file, s3_key)

        # Obtener tamaño
        size = await self.storage.get_video_size(s3_key)

        # Guardar metadata en Mongo
        video_id = await save_video_metadata(s3_key, url, size)

        return {"id": video_id, "s3_key": s3_key, "url": url, "size": size}

    async def get_video_info(self, s3_key: str):
        return await get_video_metadata(s3_key)

    async def delete_video(self, s3_key: str):
        # Eliminar de S3
        await self.storage.delete_video(s3_key)
        # Eliminar de Mongo
        deleted = await delete_video_metadata(s3_key)
        return deleted
