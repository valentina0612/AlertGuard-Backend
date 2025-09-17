from pymongo import MongoClient, server_api
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure
from .settings import settings
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class _MongoDBConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_connection()
        return cls._instance
    
    def _initialize_connection(self) -> None:
        """Inicializa la conexión con MongoDB"""
        try:
            self.client = MongoClient(
                settings.MONGODB_URI,
                server_api=server_api.ServerApi('1'),
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                retryReads=True,
                appname="AlertGuard"
            )
            self.client.admin.command('ping')
            logger.info("Conexión a MongoDB establecida")
            
            self.db = self.client[settings.MONGODB_DB_NAME]
            self.videos = self.db["videos"]
            
        except ConnectionFailure as e:
            logger.error(f"Error de conexión: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            raise

    def get_db(self) -> Database:
        return self.db

    def get_videos_collection(self):
        return self.videos

    def verificar_conexion(self) -> Tuple[bool, str]:
        try:
            self.client.admin.command('ping')
            return True, "Conexión activa"
        except Exception as e:
            return False, str(e)

# Instancia singleton interna
_internal_connection = _MongoDBConnection()

# Exporta las variables directas para importación simple
db = _internal_connection.get_db()
videos = _internal_connection.get_videos_collection()

def verificar_conexion() -> Tuple[bool, str]:
    """Función de conveniencia para verificar conexión"""
    return _internal_connection.verificar_conexion()