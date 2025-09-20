def generar_resumen(resultados):
    # Crear un set para guardar pares únicos (type, track_id)
    unique_detections = {(d["type"], d["track_id"]) for d in resultados["detections"]}

    summary = {
        "status": "completed",
        "weapons_detected": any(d["type"] == "weapon" for d in resultados["detections"]),
        "covered_faces_detected": any(d["type"] == "covered" for d in resultados["detections"]),
        "abnormal_behavior_detected": any(d["type"] == "abnormal action" for d in resultados["detections"]),
        "normal_people_detected": any(d["type"] == "normal person" for d in resultados["detections"]),
        "weapons_count":sum(1 for t, _ in unique_detections if t == "weapon"),
        "covered_count":sum(1 for t, _ in unique_detections if t == "covered"),
        "abnormal_count":sum(1 for t, _ in unique_detections if t == "abnormal action"),
        "normal_count":sum(1 for t, _ in unique_detections if t == "normal person"),
        "detections": resultados["detections"]
    }
    return summary
