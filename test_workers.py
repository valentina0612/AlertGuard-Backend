"""
Script de prueba para verificar que los workers funcionan correctamente
"""
from servicios.WorkerManager import should_process_frame

# Simular 3 workers procesando frames
print("=== Prueba de distribución de frames con 3 workers ===\n")

total_workers = 3
max_frames = 15

for worker_id in range(total_workers):
    print(f"Worker {worker_id} procesará los frames:")
    frames = []
    for frame_num in range(1, max_frames + 1):
        if should_process_frame(frame_num, worker_id, total_workers):
            frames.append(frame_num)
    print(f"  {frames}\n")

print("=== Verificación: cada frame debe ser procesado por UN solo worker ===")
all_frames = set()
for worker_id in range(total_workers):
    for frame_num in range(1, max_frames + 1):
        if should_process_frame(frame_num, worker_id, total_workers):
            if frame_num in all_frames:
                print(f"❌ ERROR: Frame {frame_num} procesado por múltiples workers")
            all_frames.add(frame_num)

if len(all_frames) == max_frames:
    print(f"✅ CORRECTO: Todos los {max_frames} frames fueron procesados exactamente una vez")
else:
    print(f"❌ ERROR: Solo {len(all_frames)}/{max_frames} frames fueron procesados")
