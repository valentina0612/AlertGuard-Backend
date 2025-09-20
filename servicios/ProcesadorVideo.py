import cv2
import numpy as np

# Configuración de video
SEQUENCE_LENGTH = 25
SKIP_FRAMES_YOLO = 2
STREAM_SIZE = (480, 270)
JPEG_QUALITY = 100


# Preprocesamiento para 3D CNN
def preprocesar_imagen(frame):
    """Redimensiona y normaliza el frame para ser usado en el modelo 3D CNN."""
    IMG_SIZE = 112
    if frame is None or frame.size == 0:
        return None
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    return frame


# Redimensionamiento para streaming
def preparar_frame(frame):
    """Escala un frame al tamaño definido para el streaming."""
    return cv2.resize(frame, STREAM_SIZE)
