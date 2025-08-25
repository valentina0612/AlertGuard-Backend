# app.py
import os
import cv2
import asyncio
import shutil
import tempfile
import numpy as np
import tensorflow as tf
from collections import deque, defaultdict
from typing import Dict
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
import base64
import json

# =========================
# Modelos
# =========================
modeloDeep = tf.keras.models.load_model("modelo3DCNN.keras")
modeloObjetos = YOLO("YOLO.pt")
print("Modelos cargados correctamente")

# =========================
# Config
# =========================
SEQUENCE_LENGTH = 25
SKIP_FRAMES_YOLO = 3     # YOLO cada 3 frames
STREAM_SIZE = (480, 270) # resolución interna
JPEG_QUALITY = 70        # calidad de salida del JPEG (más bajo = más rápido)

# =========================
# Estado por sesión
# =========================
sessions_files: Dict[str, str] = {}   # session_id -> ruta del video
alerts: Dict[str, bool] = {}          # session_id -> alerta disparada
buffers: Dict[str, deque] = {}        # session_id -> buffer de frames (para modeloDeep)
# nuevo estado
analysis_results: Dict[str, Dict] = {}  # session_id -> resultados acumulados

# =========================
# Utilidades
# =========================
def preprocesar_imagen(frame):
    IMG_SIZE = 112
    if frame is None or frame.size == 0:
        return None
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    return frame

def new_session_id() -> str:
    # id cortito para url
    return next(tempfile._get_candidate_names())

# =========================
# FastAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # o ["*"] durante pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Guarda el archivo en carpeta temporal
    tmp_dir = tempfile.mkdtemp(prefix="vid_")
    dst = os.path.join(tmp_dir, file.filename or "video.mp4")
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Crea una sesión
    sid = new_session_id()
    sessions_files[sid] = dst
    alerts[sid] = False
    buffers[sid] = deque(maxlen=SEQUENCE_LENGTH)
    analysis_results[sid] = {
    "weapons": set(),
    "covered": set(),
    "abnormal": set(),
    "normal": set(),
    "detections": []
}


    return JSONResponse({"session_id": sid})


def _cleanup_session(session_id: str, remove_file: bool = True):
    # Cierra y limpia estado y archivo temporal
    path = sessions_files.pop(session_id, None)
    alerts.pop(session_id, None)
    buffers.pop(session_id, None)
    if remove_file and path and os.path.exists(path):
        try:
            # borra el video y su carpeta temporal
            d = os.path.dirname(path)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
            else:
                os.remove(path)
        except Exception:
            pass

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if session_id not in sessions_files:
        await websocket.close(code=1008)  # Policy Violation
        return

    await websocket.accept()
    video_path = sessions_files[session_id]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        await websocket.close(code=1011)  # Internal Error
        return

    buf = buffers[session_id]
    alerta_disparada = False
    frame_count = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            frame = cv2.resize(frame, STREAM_SIZE)

            # Inicializar siempre annotated
            annotated = frame.copy()

            if frame_count % SKIP_FRAMES_YOLO == 0:
                results = modeloObjetos.track(frame, persist=True, conf=0.6)
                annotated = results[0].plot()

                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = results[0].names[cls]
                    track_id = int(box.id[0]) if box.id is not None else None

                    if track_id is not None:
                        if label == "weapon":
                            analysis_results[session_id]["weapons"].add(track_id)
                        elif label == "covered":
                            analysis_results[session_id]["covered"].add(track_id)
                        elif label == "abnormal action":
                            analysis_results[session_id]["abnormal"].add(track_id)
                        elif label == "normal person":
                            analysis_results[session_id]["normal"].add(track_id)
                        
                        if(label != "normal person"):  # Guardar todas las detecciones excepto "normal person"
                            analysis_results[session_id]["detections"].append({
                                "type": label,
                                "confidence": float(box.conf[0]),
                                "frame_number": frame_count,
                                "track_id": track_id
                            })


            # Deep solo hasta primera anomalía
            if not alerta_disparada:
                fpre = preprocesar_imagen(frame)
                if fpre is not None:
                    buf.append(fpre)

                if len(buf) == SEQUENCE_LENGTH:
                    batch = np.expand_dims(np.array(buf), axis=0)
                    pred = np.argmax(modeloDeep.predict(batch, verbose=0), axis=1)[0]
                    if pred == 1:
                        alerta_disparada = True
                        alerts[session_id] = True

            if alerta_disparada:
                cv2.putText(
                    annotated, "ADVERTENCIA", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )
                msg = {"type": "alert", "status": "warning", "message": "Anomalia detectada"}
                await websocket.send_text(json.dumps(msg))

            # Codificar frame a JPEG
            ok_enc, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok_enc:
                continue

            # Convertir a base64 para enviar por WS
            frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
            msg = {"type": "frame", "frame": frame_b64, "status":"processing"}
            await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(0.03)  # ~30 FPS
    except WebSocketDisconnect:
        print(f"Cliente desconectado: {session_id}")
    finally:
        cap.release()
        msg = {"type": "end", "status": "completed"}
        await websocket.send_text(json.dumps(msg))
        await websocket.close()
        _cleanup_session(session_id, remove_file=True)

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    if session_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Session not found")

    res = analysis_results[session_id]

    summary = {
        "status": "completed",
        "weapons_detected": len(res["weapons"]) > 0,
        "covered_faces_detected": len(res["covered"]) > 0,
        "abnormal_behavior_detected": len(res["abnormal"]) > 0,
        "normal_people_detected": len(res["normal"]) > 0,
        "weapons_count": len(res["weapons"]),
        "covered_count": len(res["covered"]),
        "abnormal_count": len(res["abnormal"]),
        "normal_count": len(res["normal"]),
        "detections": res["detections"],
        "timestamp": int(asyncio.get_event_loop().time())
    }
    return JSONResponse(summary)

# Para correr:
# uvicorn app:app --reload