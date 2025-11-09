# Guía de Implementación - WebSockets con Workers Paralelos

## 🎯 Resumen

El backend ahora soporta **múltiples conexiones WebSocket paralelas** para procesar videos más rápido. Cada "worker" procesa un subconjunto de frames del video.

## 📡 Endpoints disponibles

### 1. Endpoint con Workers (NUEVO)
```
wss://alertguard-backend-production.up.railway.app/api/ws/{session_id}/worker-{worker_id}
```

**Parámetros:**
- `session_id`: ID de la sesión (obtenido del endpoint `/upload`)
- `worker_id`: Número del worker (0, 1, 2, etc.)

### 2. Endpoint original (compatibilidad)
```
wss://alertguard-backend-production.up.railway.app/api/ws/{session_id}
```
Este endpoint sigue funcionando igual que antes (equivalente a worker-0).

## 🔧 Cómo funciona

### Distribución de frames:
- **Worker 0** procesa frames: 1, 4, 7, 10, 13...
- **Worker 1** procesa frames: 2, 5, 8, 11, 14...
- **Worker 2** procesa frames: 3, 6, 9, 12, 15...

Cada worker recibe:
- Solo los frames que le corresponden
- Los mismos mensajes de alertas
- Mensaje `"end"` cuando termina

## 💻 Implementación en el Frontend

### ⚠️ IMPORTANTE: Ordenamiento de Frames

Los frames de diferentes workers pueden llegar **desordenados**. Debes ordenarlos antes de mostrarlos para evitar que el video se vea "trabado".

### Ejemplo con 3 workers + ordenamiento:

```typescript
export class VideoService {
  private wsConnections: WebSocket[] = [];
  private readonly NUM_WORKERS = 3;
  private frameBuffer: Map<number, string> = new Map(); // Buffer para ordenar frames
  private nextFrameToShow = 1; // Próximo frame a mostrar

  connectParallel(
    sessionId: string,
    onFrame: (frameUrl: string) => void,
    onAlert: (msg: string) => void,
    onEnd: () => void
  ): void {
    let completedWorkers = 0;

    // Crear múltiples conexiones WebSocket
    for (let i = 0; i < this.NUM_WORKERS; i++) {
      const ws = new WebSocket(
        `wss://alertguard-backend-production.up.railway.app/api/ws/${sessionId}/worker-${i}`
      );

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'frame') {
            // Frame recibido con número de secuencia
            const frameNumber = data.frame_number;
            const frameData = `data:image/jpeg;base64,${data.data}`;

            // Guardar en buffer
            this.frameBuffer.set(frameNumber, frameData);

            // Mostrar frames en orden
            this.showOrderedFrames(onFrame);

          } else if (data.type === 'alert') {
            onAlert(data.message || '⚠️ Anomalía detectada');

          } else if (data.type === 'end') {
            completedWorkers++;
            console.log(`Worker ${data.worker_id} completado. Procesó ${data.frames_processed} frames`);

            // Solo llamar onEnd cuando TODOS los workers terminen
            if (completedWorkers === this.NUM_WORKERS) {
              // Mostrar frames restantes
              this.flushRemainingFrames(onFrame);
              onEnd();
            }
          }
        } catch (err) {
          console.error('Error:', err);
        }
      };

      ws.onerror = (error) => {
        console.error(`Error en worker ${i}:`, error);
      };

      this.wsConnections.push(ws);
    }
  }

  // Mostrar frames en orden secuencial
  private showOrderedFrames(onFrame: (frameUrl: string) => void): void {
    while (this.frameBuffer.has(this.nextFrameToShow)) {
      const frame = this.frameBuffer.get(this.nextFrameToShow)!;
      onFrame(frame);
      this.frameBuffer.delete(this.nextFrameToShow);
      this.nextFrameToShow++;
    }
  }

  // Mostrar frames restantes al final
  private flushRemainingFrames(onFrame: (frameUrl: string) => void): void {
    const sortedFrames = Array.from(this.frameBuffer.entries())
      .sort((a, b) => a[0] - b[0]);

    for (const [_, frameData] of sortedFrames) {
      onFrame(frameData);
    }

    this.frameBuffer.clear();
  }

  // Cerrar todas las conexiones
  disconnect(): void {
    this.wsConnections.forEach(ws => ws.close());
    this.wsConnections = [];
    this.frameBuffer.clear();
    this.nextFrameToShow = 1;
  }
}
```

## 📊 Mensajes que recibirás

### 1. Mensaje de inicio (cada worker)
```json
{
  "type": "start",
  "status": "processing",
  "message": "Worker 0 procesando..."
}
```

### 2. Frame (con número de secuencia)
```json
{
  "type": "frame",
  "frame_number": 42,
  "worker_id": 1,
  "data": "/9j/4AAQSkZJRgABAQAA..." // Base64 del JPEG
}
```

**IMPORTANTE:** Usa `frame_number` para ordenar los frames antes de mostrarlos.

### 3. Alerta (cuando se detecta anomalía)
```json
{
  "type": "alert",
  "status": "warning",
  "message": "Anomalía detectada"
}
```

### 4. Fin (cuando un worker termina)
```json
{
  "type": "end",
  "status": "completed",
  "worker_id": 0,
  "frames_processed": 250
}
```

## ⚙️ Configuración recomendada

### Número de workers según el caso:

- **1 worker**: Videos cortos (<30 seg) o conexión lenta
- **2 workers**: Videos medianos (30-60 seg)
- **3 workers**: Videos largos (>60 seg) - **RECOMENDADO**
- **4+ workers**: Solo si el servidor y cliente tienen recursos suficientes

### Importante:
- Cada worker consume ~100-150MB de RAM en el cliente
- El servidor puede manejar hasta 5 workers por sesión
- Más workers NO siempre significa más velocidad (depende del CPU del servidor)

## 🧪 Pruebas

Para probar que funciona:

1. Subir un video y obtener el `session_id`
2. Conectar 3 workers con diferentes `worker_id` (0, 1, 2)
3. Verificar que recibes frames en las 3 conexiones
4. Esperar a que los 3 workers envíen `"type": "end"`
5. Cerrar todas las conexiones

## 🔄 Compatibilidad con código anterior

Si no quieres usar workers paralelos, puedes seguir usando:

```typescript
// Esto sigue funcionando igual que antes
const ws = new WebSocket(`wss://.../api/ws/${sessionId}`);
```

## 🐛 Troubleshooting

**Problema:** No recibo frames
- Verifica que todos los workers tengan el mismo `session_id`
- Revisa la consola del navegador para errores de conexión

**Problema:** Los workers no terminan
- Asegúrate de estar esperando a que TODOS los workers envíen `"type": "end"`
- Verifica que no haya workers desconectados prematuramente

**Problema:** Rendimiento no mejora
- Reduce el número de workers a 2
- Verifica la latencia de red
- Revisa el uso de CPU/RAM del servidor

## 📞 Soporte

Si tienes problemas, verifica los logs del backend con:
```bash
grep "Worker" logs.txt
```

Verás mensajes como:
```
✅ Worker 0 conectado para sesión abc123 (Total workers: 3)
✅ Todos los workers completados para sesión abc123
```
