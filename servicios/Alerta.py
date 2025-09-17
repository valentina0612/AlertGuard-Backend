alerts = {}

def activar_alerta(session_id, mensaje="Anomalia detectada"):
    alerts[session_id] = True
    return {"type": "alert", "status": "warning", "message": mensaje}

def limpiar_alerta(session_id):
    alerts[session_id] = False
