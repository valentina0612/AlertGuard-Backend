import tempfile
import os
import shutil
import time
from collections import deque
from typing import Dict

# Estado de sesiones
finished_sessions: Dict[str, float] = {}  # session_id -> timestamp
sessions_files: Dict[str, str] = {}
alerts: Dict[str, bool] = {}
buffers: Dict[str, deque] = {}
analysis_results: Dict[str, Dict] = {}

# =========================
# Generar nuevo ID de sesión
# =========================
def new_session_id() -> str:
    return next(tempfile._get_candidate_names())

# =========================
# Crear sesión nueva
# =========================
def create_session(session_id: str, video_path: str):
    sessions_files[session_id] = video_path
    alerts[session_id] = False
    buffers[session_id] = deque(maxlen=25)  # SEQUENCE_LENGTH
    analysis_results[session_id] = {
        "detections": []
    }

# =========================
# Eliminar sesión
# =========================
def cleanup_session(session_id: str, remove_file: bool = True, keep_results: bool = True):
    """Limpia buffers, archivos y alertas. 
    Si keep_results=True, conserva resultados hasta consulta en /results."""
    path = sessions_files.pop(session_id, None)
    alerts.pop(session_id, None)
    buffers.pop(session_id, None)

    if not keep_results:
        analysis_results.pop(session_id, None)

    if remove_file and path and os.path.exists(path):
        try:
            d = os.path.dirname(path)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
            else:
                os.remove(path)
        except Exception:
            pass

    # Registrar fin de sesión
    finished_sessions[session_id] = time.time()
