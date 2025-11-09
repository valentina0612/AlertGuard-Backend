def generar_resumen(resultados):
    # Crear un set para guardar pares únicos (type, track_id)
    unique_detections = {(d["type"], d["track_id"]) for d in resultados["detections"]}

    summary = {
        "status": "completed",
        "weapons_detected": any(d["type"] == "Arma" for d in resultados["detections"]),
        "covered_faces_detected": any(d["type"] == "Cubierto" for d in resultados["detections"]),
        "abnormal_behavior_detected": any(d["type"] == "Accion abnormal" for d in resultados["detections"]),
        "normal_people_detected": any(d["type"] == "Persona normal" for d in resultados["detections"]),
        "weapons_count":sum(1 for t, _ in unique_detections if t == "Arma"),
        "covered_count":sum(1 for t, _ in unique_detections if t == "Cubierto"),
        "abnormal_count":sum(1 for t, _ in unique_detections if t == "Accion abnormal"),
        "normal_count":sum(1 for t, _ in unique_detections if t == "Persona normal"),
        "detections": resultados["detections"]
    }
    return summary
