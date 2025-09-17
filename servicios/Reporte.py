def generar_resumen(resultados):

    summary = {
        "status": "completed",
        "weapons_detected": any(d["type"] == "weapon" for d in resultados["detections"]),
        "covered_faces_detected": any(d["type"] == "covered" for d in resultados["detections"]),
        "abnormal_behavior_detected": any(d["type"] == "abnormal action" for d in resultados["detections"]),
        "normal_people_detected": any(d["type"] == "normal person" for d in resultados["detections"]),
        "weapons_count": sum(1 for d in resultados["detections"] if d["type"] == "weapon"),
        "covered_count": sum(1 for d in resultados["detections"] if d["type"] == "covered"),
        "abnormal_count": sum(1 for d in resultados["detections"] if d["type"] == "abnormal action"),
        "normal_count": sum(1 for d in resultados["detections"] if d["type"] == "normal person"),
        "detections": resultados["detections"]
    }
    return summary
