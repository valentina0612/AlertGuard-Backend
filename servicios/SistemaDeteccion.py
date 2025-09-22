import asyncio
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# =========================
# Cargar modelos
# =========================
modeloDeep = tf.keras.models.load_model("./modelosIA/modelo3DCNN.keras", compile=False)
modeloObjetos = YOLO("./modelosIA/best.pt", verbose=False)
print("Modelos cargados correctamente")

# =========================
# Cola y Executor
# =========================
cnn_queue: asyncio.Queue = asyncio.Queue()
executor = ThreadPoolExecutor(max_workers=2)

# =========================
# Predicción con CNN
# =========================
def _predict_batch(batch):
    return int(np.argmax(modeloDeep.predict(batch, verbose=0), axis=1)[0])

async def run_cnn_batch(batch):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _predict_batch, batch)

# =========================
# Detección con YOLO
# =========================
def analizar_con_modelos(frame, results_dict, frame_count):
    results = modeloObjetos.track(frame, persist=True, conf=0.35, imgsz=288,verbose=False, classes=[0, 1, 2])
    annotated = results[0].plot()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        track_id = int(box.id[0]) if box.id is not None else None

        if track_id is not None and label != "Persona normal":
            results_dict["detections"].append({
                "type": label,
                "confidence": float(box.conf[0]),
                "frame_number": frame_count,
                "track_id": track_id
            })
    return annotated
