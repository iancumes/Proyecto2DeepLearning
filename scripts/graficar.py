"""Genera las figuras de la sección 2.3 (iteraciones) a partir de resultados/<run>/.

Para cada carpeta de resultados/ con un final.json (o al menos un checkpoint):
- <run>_entrenamiento.png: recompensa cruda por episodio (monitor) + media móvil
  (rollout/ep_rew_mean de TensorBoard).
- <run>_evaluacion.png: recompensa de evaluación greedy (media +/- desviación) vs pasos.
- <run>_diagnostico.png: pérdidas/entropía/KL/clip (PPO) o pérdida/epsilon (DQN).
- <run>_divergencia.png: monitor de valores V(s)/max Q(s,a) sobre estados fijos.

Y comparaciones combinadas entre iteraciones:
- comparacion_reward_vs_pasos.png, comparacion_evaluacion_vs_pasos.png.

Uso:
    python scripts/graficar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "entregables" / "figuras"

ETIQUETAS = {
    "iter1_dqn": "Iter 1 - DQN",
    "iter2_ppo_v5_directo": "Iter 2 - PPO (v5 directo)",
    "iter3_ppo_final": "Iter 3 - PPO (final)",
}
COLORES = {
    "iter1_dqn": "#c0392b",
    "iter2_ppo_v5_directo": "#f39c12",
    "iter3_ppo_final": "#2980b9",
}


def cargar_escalares_tb(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Lee todos los eventos de TensorBoard de un run y devuelve {tag: (pasos, valores)}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    eventos = sorted((run_dir / "tb").glob("*/events.out.tfevents.*"))
    acumulado: dict[str, list[tuple[int, float]]] = {}
    for ruta_evento in eventos:
        ea = EventAccumulator(str(ruta_evento), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            acumulado.setdefault(tag, [])
            for evento in ea.Scalars(tag):
                acumulado[tag].append((evento.step, evento.value))
    salida = {}
    for tag, puntos in acumulado.items():
        puntos.sort(key=lambda p: p[0])
        pasos = np.array([p[0] for p in puntos])
        valores = np.array([p[1] for p in puntos])
        salida[tag] = (pasos, valores)
    return salida


def cargar_monitor(run_dir: Path) -> pd.DataFrame | None:
    from stable_baselines3.common.monitor import load_results

    carpeta = run_dir / "monitor"
    if not carpeta.is_dir() or not any(carpeta.glob("*monitor.csv")):
        return None
    try:
        return load_results(str(carpeta))
    except Exception:
        return None


def cargar_evaluaciones(run_dir: Path) -> dict[str, np.ndarray] | None:
    ruta = run_dir / "eval" / "evaluations.npz"
    if not ruta.is_file():
        return None
    datos = np.load(ruta)
    return {clave: datos[clave] for clave in datos.files}


def graficar_entrenamiento(run_id: str, run_dir: Path) -> None:
    monitor = cargar_monitor(run_dir)
    escalares = cargar_escalares_tb(run_dir)

    fig, eje = plt.subplots(figsize=(7, 4.5))
    if monitor is not None and len(monitor) > 0:
        eje.scatter(monitor["t"] if "t" in monitor else range(len(monitor)), monitor["r"], s=4, alpha=0.25, color="gray", label="Episodios (recompensa cruda)")
    if "rollout/ep_rew_mean" in escalares:
        pasos, valores = escalares["rollout/ep_rew_mean"]
        eje.plot(pasos, valores, color=COLORES.get(run_id, "#2980b9"), linewidth=2, label="Media móvil (100 ep., TensorBoard)")
    eje.set_xlabel("Paso de entrenamiento (o tiempo, según eje disponible)")
    eje.set_ylabel("Recompensa por episodio (recortada en entrenamiento)")
    eje.set_title(f"Entrenamiento: {ETIQUETAS.get(run_id, run_id)}")
    eje.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / f"{run_id}_entrenamiento.png", dpi=130)
    plt.close(fig)


def graficar_evaluacion(run_id: str, run_dir: Path) -> None:
    evals = cargar_evaluaciones(run_dir)
    if evals is None:
        return
    pasos = evals["timesteps"]
    resultados = evals["results"]
    media = resultados.mean(axis=1)
    desv = resultados.std(axis=1)

    fig, eje = plt.subplots(figsize=(7, 4.5))
    eje.plot(pasos, media, color=COLORES.get(run_id, "#2980b9"), linewidth=2, marker="o")
    eje.fill_between(pasos, media - desv, media + desv, color=COLORES.get(run_id, "#2980b9"), alpha=0.2)
    eje.set_xlabel("Pasos de entrenamiento")
    eje.set_ylabel("Recompensa de evaluación (greedy, cruda, 5 episodios)")
    eje.set_title(f"Evaluación durante el entrenamiento: {ETIQUETAS.get(run_id, run_id)}")
    fig.tight_layout()
    fig.savefig(FIGURAS / f"{run_id}_evaluacion.png", dpi=130)
    plt.close(fig)


def graficar_diagnostico(run_id: str, run_dir: Path, algoritmo: str) -> None:
    escalares = cargar_escalares_tb(run_dir)
    if not escalares:
        return

    if algoritmo == "ppo":
        tags = [
            ("train/value_loss", "Pérdida de valor"),
            ("train/entropy_loss", "Pérdida de entropía"),
            ("train/approx_kl", "KL aproximado"),
            ("train/clip_fraction", "Fracción recortada (PPO clip)"),
        ]
    else:
        tags = [
            ("train/loss", "Pérdida (Huber)"),
            ("rollout/exploration_rate", "Epsilon de exploración"),
        ]

    disponibles = [(tag, titulo) for tag, titulo in tags if tag in escalares]
    if not disponibles:
        return
    fig, ejes = plt.subplots(1, len(disponibles), figsize=(4.2 * len(disponibles), 3.8))
    if len(disponibles) == 1:
        ejes = [ejes]
    for eje, (tag, titulo) in zip(ejes, disponibles):
        pasos, valores = escalares[tag]
        eje.plot(pasos, valores, color=COLORES.get(run_id, "#2980b9"))
        eje.set_title(titulo)
        eje.set_xlabel("Pasos")
    fig.suptitle(f"Diagnóstico de entrenamiento: {ETIQUETAS.get(run_id, run_id)}")
    fig.tight_layout()
    fig.savefig(FIGURAS / f"{run_id}_diagnostico.png", dpi=130)
    plt.close(fig)


def graficar_divergencia(run_id: str, run_dir: Path) -> None:
    ruta = run_dir / "valores.csv"
    if not ruta.is_file():
        return
    df = pd.read_csv(ruta)
    if df.empty:
        return
    fig, eje = plt.subplots(figsize=(7, 4))
    eje.plot(df["pasos"], df["valor_medio"], color=COLORES.get(run_id, "#2980b9"))
    eje.set_xlabel("Pasos de entrenamiento")
    eje.set_ylabel("Valor medio sobre 500 estados fijos")
    eje.set_title(f"Monitor de divergencia (V/Q): {ETIQUETAS.get(run_id, run_id)}")
    eje.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIGURAS / f"{run_id}_divergencia.png", dpi=130)
    plt.close(fig)


def graficar_comparaciones(runs_presentes: list[str]) -> None:
    fig, eje = plt.subplots(figsize=(7.5, 5))
    for run_id in runs_presentes:
        escalares = cargar_escalares_tb(RESULTADOS / run_id)
        if "rollout/ep_rew_mean" in escalares:
            pasos, valores = escalares["rollout/ep_rew_mean"]
            eje.plot(pasos, valores, label=ETIQUETAS.get(run_id, run_id), color=COLORES.get(run_id))
    eje.set_xlabel("Pasos de entrenamiento")
    eje.set_ylabel("Recompensa media (100 ep., recortada)")
    eje.set_title("Comparación entre iteraciones: entrenamiento")
    eje.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "comparacion_reward_vs_pasos.png", dpi=130)
    plt.close(fig)

    fig2, eje2 = plt.subplots(figsize=(7.5, 5))
    for run_id in runs_presentes:
        evals = cargar_evaluaciones(RESULTADOS / run_id)
        if evals is not None:
            media = evals["results"].mean(axis=1)
            eje2.plot(evals["timesteps"], media, marker="o", label=ETIQUETAS.get(run_id, run_id), color=COLORES.get(run_id))
    eje2.set_xlabel("Pasos de entrenamiento")
    eje2.set_ylabel("Recompensa de evaluación greedy (cruda)")
    eje2.set_title("Comparación entre iteraciones: evaluación greedy")
    eje2.legend()
    fig2.tight_layout()
    fig2.savefig(FIGURAS / "comparacion_evaluacion_vs_pasos.png", dpi=130)
    plt.close(fig2)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    runs_presentes = []
    for run_id in ETIQUETAS:
        run_dir = RESULTADOS / run_id
        if not run_dir.is_dir():
            continue
        runs_presentes.append(run_id)
        algoritmo = "dqn" if "dqn" in run_id else "ppo"
        print(f"Graficando {run_id}...")
        try:
            graficar_entrenamiento(run_id, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  aviso entrenamiento: {exc}")
        try:
            graficar_evaluacion(run_id, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  aviso evaluacion: {exc}")
        try:
            graficar_diagnostico(run_id, run_dir, algoritmo)
        except Exception as exc:  # noqa: BLE001
            print(f"  aviso diagnostico: {exc}")
        try:
            graficar_divergencia(run_id, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  aviso divergencia: {exc}")

    if runs_presentes:
        print("Graficando comparaciones...")
        graficar_comparaciones(runs_presentes)
    print(f"Listo. Figuras en {FIGURAS}")


if __name__ == "__main__":
    main()
