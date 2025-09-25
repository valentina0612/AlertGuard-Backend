import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
# Importar el controlador principal
from ControladorApi import app as controlador_app

# =========================
# Crear aplicación principal
# =========================
app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar controlador en /api (para que no choque con la raíz /)
app.mount("/api", controlador_app)

# Ruta raíz de prueba
@app.get("/")
def root():
    return {"message": "Backend funcionando!"}

# =========================
# Punto de entrada
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render asigna el puerto aquí
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

