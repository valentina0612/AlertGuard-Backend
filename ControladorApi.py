import asyncio
import shutil
import os
import tempfile
import cv2
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Importar servicios
from servicios.SesionUsuario import (
    new_session_id, create_session, cleanup_session,
    sessions_files, alerts, buffers, analysis_results
)
from servicios.SistemaDeteccion import cnn_queue, run_cnn_batch, analizar_con_modelos
from servicios.ProcesadorVideo import preprocesar_imagen, STREAM_SIZE, JPEG_QUALITY, SKIP_FRAMES_YOLO
from servicios.Reporte import generar_resumen
from config.storage import StorageService
from servicios.mongo import save_video_metadata  # lo que definimos para Mongo
from config.settings import settings

storage_service = StorageService()
# --- Configuración de FastAPI ---
app = FastAPI()

# --- Executor para procesos pesados ---
NUM_CNN_WORKERS = 1
TEMP_FILE_PREFIX = "video_processing_"
executor = ProcessPoolExecutor(max_workers=NUM_CNN_WORKERS)

# --- Parámetros de la secuencia ---
SEQUENCE_LENGTH = 25  # Ventana máxima
MIN_SEQUENCE = 8      # Mínimos frames para primera inferencia
cnn_busy = {}         # session_id -> bool para controlar inferencias en vuelo

# Wrapper síncrono para ejecutar CNN
def run_cnn_batch_sync(batch):
    result = run_cnn_batch(batch)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result

async def run_cnn_async(batch):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, run_cnn_batch_sync, batch)

# --- Worker CNN ---
async def cnn_worker():
    while True:
        try:
            batch, websocket, session_id = await cnn_queue.get()
            try:
                pred = await run_cnn_async(batch)
                if pred == 1 and not alerts.get(session_id, False):
                    alerts[session_id] = True
                    msg = {"type": "alert", "status": "warning", "message": "Anomalía detectada"}
                    try:
                        await websocket.send_text(json.dumps(msg))
                    except Exception:
                        pass  # websocket cerrado
            finally:
                cnn_busy[session_id] = False
        except Exception:
            pass
        finally:
            cnn_queue.task_done()

# --- Lanzar los workers CNN ---
def launch_cnn_workers():
    for i in range(NUM_CNN_WORKERS):
        asyncio.create_task(cnn_worker(), name=f"cnn_worker_{i}")

# --- Endpoint: Subida de video ---
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Guardar temporalmente en disco (para tu pipeline CNN/YOLO)
    tmp_dir = tempfile.mkdtemp(prefix="vid_")
    dst = os.path.join(tmp_dir, file.filename or "video.mp4")
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Crear sesión local
    sid = new_session_id()
    create_session(sid, dst)
    buffers[sid] = deque(maxlen=SEQUENCE_LENGTH)
    cnn_busy[sid] = False

    # Subir a S3
    s3_key = f"videos/{sid}_{file.filename}"
    with open(dst, "rb") as f:
        s3_key, url = await storage_service.upload_to_s3(f, s3_key)

    # Obtener tamaño real en S3
    size = await storage_service.get_video_size(s3_key)

    # Guardar metadata en Mongo
    video_id = await save_video_metadata(s3_key, url, size)

    return JSONResponse({
        "session_id": sid,
        "video_id": video_id,
        "s3_key": s3_key,
        "url": url,
        "size": size
    })

# --- Endpoint: WebSocket ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if session_id not in sessions_files:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    video_path = sessions_files[session_id]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        await websocket.close(code=1011)
        return

    # Lanzar workers CNN si no están en ejecución
    if not any(t.get_name().startswith("cnn_worker_") for t in asyncio.all_tasks()):
        launch_cnn_workers()

    buf = buffers[session_id]
    frame_count = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            frame = cv2.resize(frame, STREAM_SIZE)
            annotated = frame.copy()

            # Detección con modelos (YOLO)
            if frame_count % SKIP_FRAMES_YOLO == 0:
                annotated = analizar_con_modelos(
                    frame, analysis_results[session_id], frame_count
                )

            # Preprocesamiento y envío a CNN (ventana deslizante)
            if not alerts[session_id]:
                fpre = preprocesar_imagen(frame)
                if fpre is not None:
                    buf.append(fpre)

                if len(buf) >= MIN_SEQUENCE and not cnn_busy.get(session_id, False):
                    try:
                        seq = np.stack(list(buf), axis=0).astype(np.float32)
                        batch = np.expand_dims(seq, axis=0)
                        cnn_busy[session_id] = True
                        await cnn_queue.put((batch, websocket, session_id))
                    except Exception:
                        pass  # saltar si algo falla

            # Mostrar alerta
            if alerts[session_id]:
                cv2.putText(
                    annotated, "ADVERTENCIA", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )

            # Enviar frame
            ok_enc, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok_enc:
                await websocket.send_bytes(jpeg.tobytes())

            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        msg = {"type": "end", "status": "completed"}
        await websocket.send_text(json.dumps(msg))
        await websocket.close()
        cleanup_session(session_id, remove_file=True)

# --- Endpoint: Resultados ---
@app.get("/results/{session_id}")
async def get_results(session_id: str):
    if session_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Session not found")

    res = analysis_results[session_id]
    summary = generar_resumen(res)
    return JSONResponse(summary)