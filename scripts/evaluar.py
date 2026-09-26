"""Evalúa un modelo entrenado con política greedy: el script del día de la presentación.

Carga los pesos y el contrato de entorno guardados junto al modelo, ejecuta N
episodios completos (sin exploración, recompensa cruda) con las funciones del
Laboratorio 5 (``ejecutar_episodio``) y reporta recompensa media, desviación y el
máximo (la métrica de la competencia). Sin pasos manuales adicionales.

Uso:
    python scripts/evaluar.py --modelo modelos/ppo_space_invaders_final.zip --episodios 5 --seed 23236
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from agente import cargar_modelo, politica_greedy  # noqa: E402
from agentes_ale import ejecutar_episodio  # noqa: E402
from entorno import crear_entorno_evaluacion  # noqa: E402


def evaluar(ruta_modelo: str, episodios: int, seed: int) -> dict:
    modelo, cfg_entorno = cargar_modelo(ruta_modelo)
    politica = politica_greedy(modelo)
    env = crear_entorno_evaluacion(cfg_entorno)

    resultados = []
    try:
        for i in range(episodios):
            semilla = seed + i
            t0 = time.time()
            resultado = ejecutar_episodio(env, politica, max_steps=None, seed=semilla)
            resultado["semilla"] = semilla
            resultado["duracion_segundos"] = round(time.time() - t0, 1)
            resultados.append(resultado)
            print(
                f"  episodio {i + 1}/{episodios} (semilla {semilla}): "
                f"recompensa={resultado['recompensa_total']:.0f} pasos={resultado['pasos']} "
                f"({resultado['duracion_segundos']:.1f}s)"
            )
    finally:
        env.close()

    puntajes = [r["recompensa_total"] for r in resultados]
    resumen = {
        "modelo": str(Path(ruta_modelo).resolve()),
        "episodios": episodios,
        "seed_inicial": seed,
        "puntajes": puntajes,
        "recompensa_media": statistics.mean(puntajes),
        "recompensa_desviacion": statistics.pstdev(puntajes) if len(puntajes) > 1 else 0.0,
        "recompensa_maxima": max(puntajes),
        "resultados_detallados": resultados,
    }
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23236)
    parser.add_argument("--salida", default=None, help="JSON de salida (por defecto resultados/evaluaciones/<nombre>.json)")
    args = parser.parse_args()

    print(f"Evaluando {args.modelo} con politica greedy, {args.episodios} episodios (seed base {args.seed})...")
    resumen = evaluar(args.modelo, args.episodios, args.seed)

    print("\n--- Resumen ---")
    print(f"Puntajes: {resumen['puntajes']}")
    print(f"Media: {resumen['recompensa_media']:.1f}  Desv: {resumen['recompensa_desviacion']:.1f}  Maximo: {resumen['recompensa_maxima']:.0f}")

    nombre_modelo = Path(args.modelo).stem
    ruta_salida = Path(args.salida) if args.salida else RAIZ / "resultados" / "evaluaciones" / f"{nombre_modelo}.json"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {ruta_salida}")


if __name__ == "__main__":
    main()
