"""Genera el video entregable: mismo protocolo de evaluación (5 episodios, greedy),
se conserva el de mayor puntaje como el .mp4 oficial, con manifiesto reproducible
(sha256, fotogramas, fps, duración) igual que en el Laboratorio 5.

Uso:
    python scripts/generar_video.py --modelo modelos/ppo_space_invaders_final.zip --seed 23236
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from agente import cargar_modelo, politica_greedy  # noqa: E402
from agentes_ale import generar_video_agente  # noqa: E402
from entorno import envolver  # noqa: E402

ENTREGABLES = RAIZ / "entregables"
VIDEOS_TMP = RAIZ / "resultados" / "videos_tmp"
VIDEO_FINAL = ENTREGABLES / "space_invaders_ppo_final.mp4"
METRICAS = ENTREGABLES / "metricas_video.json"


def inspeccionar_video(ruta: Path) -> dict[str, int | float]:
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise RuntimeError(f"No fue posible abrir el video: {ruta}")
    cuadros = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(captura.get(cv2.CAP_PROP_FPS))
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    captura.release()
    if cuadros <= 0 or fps <= 0 or ancho <= 0 or alto <= 0:
        raise RuntimeError("El archivo MP4 no contiene propiedades de video validas")
    return {
        "cuadros_video": cuadros,
        "fps": fps,
        "duracion_segundos": round(cuadros / fps, 1),
        "ancho": ancho,
        "alto": alto,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23236)
    args = parser.parse_args()

    ENTREGABLES.mkdir(parents=True, exist_ok=True)
    VIDEOS_TMP.mkdir(parents=True, exist_ok=True)

    modelo, cfg_entorno = cargar_modelo(args.modelo)
    politica = politica_greedy(modelo)

    resultados = generar_video_agente(
        cfg_entorno["env_id"],
        politica,
        VIDEOS_TMP,
        "space_invaders_ppo",
        n_episodios=args.episodios,
        seed=args.seed,
        max_steps=None,
        envolver=lambda env: envolver(env, cfg_entorno, entrenamiento=False),
        **cfg_entorno["make_kwargs"],
    )

    mejor = max(resultados, key=lambda r: r["recompensa_total"])
    fuente = Path(mejor["video"])
    shutil.copy2(fuente, VIDEO_FINAL)
    inspeccion = inspeccionar_video(VIDEO_FINAL)
    sha256 = hashlib.sha256(VIDEO_FINAL.read_bytes()).hexdigest()

    manifiesto = {
        "autor": "Ian Cumes",
        "carne": "23236",
        "modelo": str(Path(args.modelo).resolve()),
        "entorno": cfg_entorno,
        "protocolo_evaluacion": "5 episodios greedy (deterministic=True), recompensa cruda, semillas consecutivas",
        "puntajes_5_episodios": [r["recompensa_total"] for r in resultados],
        "episodio_elegido": {k: v for k, v in mejor.items() if k != "video"},
        "video": {
            "archivo": VIDEO_FINAL.relative_to(RAIZ).as_posix(),
            "tamano_bytes": VIDEO_FINAL.stat().st_size,
            "sha256": sha256,
            **inspeccion,
        },
        "versiones": {
            "python": platform.python_version(),
            "gymnasium": version("gymnasium"),
            "ale-py": version("ale-py"),
            "stable-baselines3": version("stable-baselines3"),
            "moviepy": version("moviepy"),
            "imageio-ffmpeg": version("imageio-ffmpeg"),
            "opencv-python": version("opencv-python"),
        },
    }
    METRICAS.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    print(f"\nVideo final: {VIDEO_FINAL} (episodio semilla {mejor['semilla']}, recompensa {mejor['recompensa_total']:.0f})")


if __name__ == "__main__":
    main()
