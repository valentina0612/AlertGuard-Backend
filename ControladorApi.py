import asyncio
import shutil
import os
import tempfile
import base64
from typing import Dict
import cv2
import json
import numpy as np
import logging  # ✅ Añadir logging
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
from servicios.mongo import save_video_metadata, update_video_results
from config.settings import settings
from servicios.WorkerManager import (
    register_worker, mark_worker_completed, get_active_workers_count,
    cleanup_workers, should_process_frame
)

# ✅ Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Mapeo global de session_id -> video_id
sessions_metadata: Dict[str, str] = {}
# Lock para sincronizar escrituras concurrentes en analysis_results
analysis_lock = asyncio.Lock()

# Wrapper síncrono para ejecutar CNN
def run_cnn_batch_sync(batch):
    result = run_cnn_batch(batch)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result

async def run_cnn_async(batch):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, run_cnn_batch_sync, batch)


# --- Worker CNN adaptado a ONNX con batch ---
async def cnn_worker():
    while True:
        try:
            batch, websocket, session_id = await cnn_queue.get()
            try:
                # Si la sesión ya tiene alerta, no volver a procesar
                if alerts.get(session_id, False):
                    cnn_busy[session_id] = False
                    continue

                pred = await run_cnn_async(batch)  # pred es un array, ej: [0,1,1,0,1]
                # Disparar alerta si la mayoría son 1 y si se ha comportamiento anormal
                abnormal_count = sum(
                1 for d in analysis_results[session_id]["detections"]
                if d["type"] == "Accion anormal" and d["confidence"] >= 0.65
                )

            # Condición de disparo
                if (
                    np.sum(pred == 1) > len(pred) / 2
                    and abnormal_count >= 2
                    and not alerts.get(session_id, False)
                ):
                    alerts[session_id] = True
                    msg = {
                        "type": "alert",
                        "status": "warning",
                        "message": "Anomalía detectada"
                    }
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
    logger.info(f"📤 Iniciando upload del archivo: {file.filename}")
    
    # Validar tipo de archivo
    if not file.content_type or not file.content_type.startswith('video/'):
        logger.warning(f"❌ Tipo de archivo no permitido: {file.content_type}")
        raise HTTPException(status_code=400, detail="Solo se permiten archivos de video")

    # Guardar temporalmente en disco (para pipeline CNN/YOLO)
    tmp_dir = tempfile.mkdtemp(prefix="vid_")
    dst = os.path.join(tmp_dir, file.filename or "video.mp4")
    
    try:
        with open(dst, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Archivo temporal guardado correctamente")
    except Exception as e:
        logger.error(f"Error guardando archivo temporal: {str(e)}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error guardando archivo temporal: {str(e)}")

    # Crear sesión local
    sid = new_session_id()
    create_session(sid, dst)
    buffers[sid] = deque(maxlen=SEQUENCE_LENGTH)
    cnn_busy[sid] = False
    
    try:
        # Subir a AWS S3
        filename = file.filename or "video.mp4"
        s3_key = f"videos/{sid}_{filename}"
        logger.info(f"☁️  Subiendo a S3")
        
        with open(dst, "rb") as f:
            s3_key, url = await storage_service.upload_to_s3(f, s3_key)
        logger.info(f"✅ Video subido a S3: {url}")

        # Obtener tamaño real en S3
        size = await storage_service.get_video_size(s3_key)

        # Guardar metadata en MongoDB (✅ SIN await - función síncrona)
        logger.info("💾 Guardando metadata en MongoDB")
        video_id = save_video_metadata(s3_key, url, size)
        sessions_metadata[sid] = video_id

        return JSONResponse({
            "session_id": sid,
            "video_id": video_id,
            "s3_key": s3_key,
            "url": url,
            "size": size,
            "filename": filename,
            "message": "Video subido y procesado correctamente"
        })

    except HTTPException as he:
        logger.error(f"❌ HTTPException durante upload: {he.detail}")
        cleanup_session(sid, remove_file=True)
        raise he
    except Exception as e:
        logger.error(f"❌ Error inesperado durante upload: {str(e)}", exc_info=True)
        cleanup_session(sid, remove_file=True)
        raise HTTPException(status_code=500, detail=f"Error procesando video: {str(e)}")

# --- Endpoint: WebSocket (con soporte para workers paralelos) ---
@app.websocket("/ws/{session_id}/worker-{worker_id}")
async def websocket_worker_endpoint(websocket: WebSocket, session_id: str, worker_id: int):
    """
    Endpoint WebSocket con soporte para múltiples workers en paralelo.
    Cada worker procesa un subconjunto de frames del video.
    """
    if session_id not in sessions_files:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # Registrar este worker
    await register_worker(session_id, worker_id)
    total_workers = await get_active_workers_count(session_id)

    logger.info(f"✅ Worker {worker_id} conectado para sesión {session_id} (Total workers: {total_workers})")

    video_path = sessions_files[session_id]
    # Cada worker necesita su propio VideoCapture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        await websocket.close(code=1011)
        return

    msg = {"type": "start", "status": "processing", "message": f"Worker {worker_id} procesando..."}
    try:
        await websocket.send_text(json.dumps(msg))
    except Exception:
        await websocket.close(code=1011)
        return

    # Lanzar workers CNN si no están en ejecución
    if not any(t.get_name().startswith("cnn_worker_") for t in asyncio.all_tasks()):
        launch_cnn_workers()

    buf = buffers[session_id]
    frame_count = 0
    frames_processed_by_worker = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1

            # Verificar si este worker debe procesar este frame
            if not should_process_frame(frame_count, worker_id, total_workers):
                continue  # Saltar este frame, le toca a otro worker

            frames_processed_by_worker += 1
            frame = cv2.resize(frame, STREAM_SIZE)
            annotated = frame.copy()

            # Detección con modelos (YOLO) - solo procesar según SKIP_FRAMES_YOLO
            if frame_count % SKIP_FRAMES_YOLO == 0:
                # Usar lock para evitar race conditions al escribir en analysis_results
                async with analysis_lock:
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

            # Enviar frame con número de secuencia para ordenamiento correcto
            ok_enc, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok_enc:
                frame_msg = {
                    "type": "frame",
                    "frame_number": frame_count,
                    "worker_id": worker_id,
                    "data": base64.b64encode(jpeg.tobytes()).decode('utf-8')
                }
                await websocket.send_text(json.dumps(frame_msg))

            # Sleep reducido para mejor rendimiento
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"🔌 Worker {worker_id} desconectado")
    except Exception as e:
        logger.error(f"❌ Error en WebSocket Worker {worker_id}: {str(e)}")
    finally:
        cap.release()

        # Marcar este worker como completado
        all_completed = await mark_worker_completed(session_id, worker_id)

        try:
            msg = {
                "type": "end",
                "status": "completed",
                "worker_id": worker_id,
                "frames_processed": frames_processed_by_worker
            }
            await websocket.send_text(json.dumps(msg))
        except:
            pass

        # Solo limpiar la sesión cuando TODOS los workers hayan terminado
        if all_completed:
            logger.info(f"✅ Todos los workers completados para sesión {session_id}")
            await cleanup_workers(session_id)
            cleanup_session(session_id, remove_file=True)

        await websocket.close()


# --- Endpoint: WebSocket (backward compatibility - sin workers) ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Endpoint WebSocket original (sin workers paralelos).
    Mantiene compatibilidad con frontend antiguo.
    """
    # Redirigir a worker-0 (comportamiento por defecto)
    await websocket_worker_endpoint(websocket, session_id, 0)

# --- Endpoint: Resultados ---
@app.get("/results/{session_id}")
async def get_results(session_id: str):
    if session_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Session not found")

    res = analysis_results[session_id]
    summary = generar_resumen(res)

    if session_id in sessions_files:
        summary["session_id"] = session_id
        summary["processed_frames"] = len(res.get("detections", []))

    return JSONResponse(summary)


@app.put("/results/{session_id}/save")
async def save_results(session_id: str):
    if session_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Session not found")

    res = analysis_results[session_id]
    summary = generar_resumen(res)

    video_id = sessions_metadata.get(session_id)

    if not video_id:
        raise HTTPException(status_code=404, detail="Video ID not found for this session")

    logger.info(f"💾 Guardando resultados en MongoDB para video {video_id}")
    success = update_video_results(video_id, summary)

    if not success:
        raise HTTPException(status_code=500, detail="Error guardando resultados en MongoDB")

    return {"message": "Resultados guardados correctamente", "video_id": video_id}




# --- Endpoint: Health Check ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "aws_s3": "configured",
            "mongodb": "configured",
            "cnn_model": "loaded",
            "yolo_model": "loaded"
        }
    }

# --- Endpoint: Info del Video ---
@app.get("/video/{session_id}/info")
async def get_video_info(session_id: str):
    if session_id not in sessions_files:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "local_path": sessions_files[session_id],
        "has_alerts": alerts.get(session_id, False),
        "buffer_size": len(buffers.get(session_id, [])),
        "analysis_results": len(analysis_results.get(session_id, {}).get("detections", []))
    }