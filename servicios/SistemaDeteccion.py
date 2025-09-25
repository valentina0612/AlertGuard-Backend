import asyncio
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# =========================
# Cargar modelos
# =========================
# Modelo ONNX
modeloDeep = ort.InferenceSession("./modelosIA/modelo3DCNN.onnx")
# Modelo YOLO
modeloObjetos = YOLO("./modelosIA/best.pt", verbose=False)
print("Modelos cargados correctamente")

# =========================
# Cola y Executor
# =========================
cnn_queue: asyncio.Queue = asyncio.Queue()
executor = ThreadPoolExecutor(max_workers=2)

# =========================
# Predicción con CNN (ONNX) para batch completo
# =========================
def _predict_batch(batch):
    """
    batch: numpy array de shape [batch_size, ...] 
    Retorna un array con las predicciones (índice de clase) de todo el batch
    """
    input_name = modeloDeep.get_inputs()[0].name
    pred_onx = modeloDeep.run(None, {input_name: batch.astype(np.float32)})[0]
    # Devuelve un array con la clase predicha para cada elemento del batch
    return np.argmax(pred_onx, axis=1)

async def run_cnn_batch(batch):
    """
    Ejecuta la predicción en un executor, retornando todas las predicciones del batch
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _predict_batch, batch)

# =========================
# Detección con YOLO
# =========================
def analizar_con_modelos(frame, results_dict, frame_count):
    results = modeloObjetos.track(frame, persist=True, conf=0.35, imgsz=288, verbose=False, classes=[0, 1, 2])
    annotated = results[0].plot()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        track_id = int(box.id[0]) if box.id is not None else None

        if track_id is not None and label != "normal person":
            results_dict["detections"].append({
                "type": label,
                "confidence": float(box.conf[0]),
                "frame_number": frame_count,
                "track_id": track_id
            })
    return annotated
