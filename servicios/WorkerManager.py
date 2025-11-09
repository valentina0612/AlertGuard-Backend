"""
Gestor de Workers para WebSockets en paralelo
Permite que múltiples WebSockets procesen el mismo video simultáneamente
"""
import asyncio
from typing import Dict, Set
from collections import defaultdict

# Estructura para trackear workers activos por sesión
active_workers: Dict[str, Set[int]] = defaultdict(set)  # session_id -> {worker_ids}
workers_completed: Dict[str, Set[int]] = defaultdict(set)  # session_id -> {worker_ids completados}
workers_lock = asyncio.Lock()  # Lock para acceso concurrente


async def register_worker(session_id: str, worker_id: int) -> bool:
    """
    Registra un nuevo worker para una sesión.
    Retorna True si se registró exitosamente.
    """
    async with workers_lock:
        active_workers[session_id].add(worker_id)
        return True


async def mark_worker_completed(session_id: str, worker_id: int) -> bool:
    """
    Marca un worker como completado.
    Retorna True si TODOS los workers de la sesión han terminado.
    """
    async with workers_lock:
        workers_completed[session_id].add(worker_id)
        # Verificar si todos los workers activos han completado
        all_completed = workers_completed[session_id] == active_workers[session_id]
        return all_completed


async def get_active_workers_count(session_id: str) -> int:
    """Retorna el número de workers activos para una sesión."""
    async with workers_lock:
        return len(active_workers[session_id])


async def cleanup_workers(session_id: str):
    """Limpia los datos de workers para una sesión."""
    async with workers_lock:
        if session_id in active_workers:
            del active_workers[session_id]
        if session_id in workers_completed:
            del workers_completed[session_id]


def should_process_frame(frame_number: int, worker_id: int, total_workers: int) -> bool:
    """
    Determina si un worker debe procesar un frame específico.

    Distribución:
    - worker-0 procesa frames 0, 3, 6, 9...
    - worker-1 procesa frames 1, 4, 7, 10...
    - worker-2 procesa frames 2, 5, 8, 11...

    Args:
        frame_number: Número de frame (1-indexed, como viene del video)
        worker_id: ID del worker (0-indexed)
        total_workers: Total de workers activos

    Returns:
        True si este worker debe procesar el frame
    """
    # Convertir frame_number a 0-indexed para el cálculo
    return (frame_number - 1) % total_workers == worker_id
