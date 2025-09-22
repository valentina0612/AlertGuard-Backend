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
    allow_origins=["http://localhost:4200"],  # frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir las rutas del controlador
app.mount("/", controlador_app)

# =========================
# Punto de entrada
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render asigna el puerto aquí
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

