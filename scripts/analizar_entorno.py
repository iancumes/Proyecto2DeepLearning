"""EDA del entorno (sección 2.1 del informe): espacios, recompensa, parpadeo, figuras.

Genera resultados/eda.json y entregables/figuras/eda_*.png.

Uso:
    python scripts/analizar_entorno.py --episodios 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from entorno import CONFIG_FINAL, envolver  # noqa: E402

gym.register_envs(ale_py)

FIGURAS = RAIZ / "entregables" / "figuras"
SALIDA_JSON = RAIZ / "resultados" / "eda.json"


def info_entorno() -> dict:
    env = gym.make(CONFIG_FINAL["env_id"], **CONFIG_FINAL["make_kwargs"])
    obs, info = env.reset(seed=0)
    datos = {
        "env_id": CONFIG_FINAL["env_id"],
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "n_acciones": int(env.action_space.n),
        "significado_acciones": list(env.unwrapped.get_action_meanings()),
        "vidas_iniciales": info.get("lives"),
        "spec_kwargs": dict(env.spec.kwargs),
        "max_num_frames_per_episode": env.spec.kwargs.get("max_num_frames_per_episode"),
    }
    env.close()
    return datos


def episodios_aleatorios(n: int, seed: int = 0) -> dict:
    env = gym.make(CONFIG_FINAL["env_id"], **CONFIG_FINAL["make_kwargs"])
    puntajes, pasos_lista, recompensas_no_cero = [], [], []
    contador_valores = Counter()
    pasos_totales = 0
    for ep in range(n):
        obs, info = env.reset(seed=seed + ep)
        vidas_previas = info.get("lives")
        terminated = truncated = False
        recompensa_total = 0.0
        pasos = 0
        ultimo_paso_con_recompensa = 0
        gaps = []
        while not (terminated or truncated):
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            recompensa_total += r
            pasos += 1
            contador_valores[float(r)] += 1
            if r != 0:
                recompensas_no_cero.append(float(r))
                gaps.append(pasos - ultimo_paso_con_recompensa)
                ultimo_paso_con_recompensa = pasos
        puntajes.append(recompensa_total)
        pasos_lista.append(pasos)
        pasos_totales += pasos
    env.close()

    pasos_con_recompensa = sum(v for k, v in contador_valores.items() if k != 0)
    pasos_sin_recompensa = contador_valores.get(0.0, 0)
    return {
        "n_episodios": n,
        "puntaje_medio": float(np.mean(puntajes)),
        "puntaje_desviacion": float(np.std(puntajes)),
        "puntaje_min": float(np.min(puntajes)),
        "puntaje_max": float(np.max(puntajes)),
        "pasos_medio": float(np.mean(pasos_lista)),
        "pasos_totales": pasos_totales,
        "pct_pasos_recompensa_cero": round(100 * pasos_sin_recompensa / pasos_totales, 1),
        "valores_recompensa_no_cero_observados": sorted({v for v in contador_valores if v != 0}),
        "distribucion_valores_recompensa": {str(k): v for k, v in sorted(contador_valores.items())},
        "puntajes_por_episodio": puntajes,
    }


def figura_frames(seed: int = 23236) -> None:
    """Frame RGB crudo vs. 84x84 en gris, y el efecto de max-pool sobre el parpadeo."""
    env_rgb = gym.make(CONFIG_FINAL["env_id"], **{**CONFIG_FINAL["make_kwargs"], "frameskip": 1})
    env_rgb.reset(seed=seed)
    frames_rgb = []
    for i in range(12):
        obs, _, term, trunc, _ = env_rgb.step(1 if i == 0 else 0)
        frames_rgb.append(obs.copy())
        if term or trunc:
            break
    env_rgb.close()

    env_final = envolver(
        gym.make(CONFIG_FINAL["env_id"], **CONFIG_FINAL["make_kwargs"]), CONFIG_FINAL, entrenamiento=False
    )
    obs_final, _ = env_final.reset(seed=seed)
    obs_final = np.array(obs_final)
    env_final.close()

    fig, ejes = plt.subplots(1, 3, figsize=(12, 4))
    ejes[0].imshow(frames_rgb[0])
    ejes[0].set_title("RGB crudo 210x160x3")
    ejes[1].imshow(obs_final[-1], cmap="gray")
    ejes[1].set_title("Preprocesado 84x84 (1 de 4 frames apilados)")
    sin_pool = np.abs(frames_rgb[1].astype(int) - frames_rgb[0].astype(int)).sum(axis=-1)
    con_pool = np.maximum(frames_rgb[1], frames_rgb[0]).astype(int)
    con_pool_diff = np.abs(con_pool - frames_rgb[0].astype(int)).sum(axis=-1)
    ejes[2].imshow(sin_pool, cmap="hot")
    ejes[2].set_title("|f(t+1)-f(t)| (evidencia de parpadeo)")
    for eje in ejes:
        eje.axis("off")
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "eda_frames.png", dpi=130)
    plt.close(fig)

    fig2, ejes2 = plt.subplots(1, 4, figsize=(12, 3.2))
    for i, eje in enumerate(ejes2):
        eje.imshow(obs_final[i], cmap="gray")
        eje.set_title(f"t-{3 - i}")
        eje.axis("off")
    fig2.suptitle("4 frames apilados (entrada real de la red)")
    fig2.tight_layout()
    fig2.savefig(FIGURAS / "eda_frame_stack.png", dpi=130)
    plt.close(fig2)


def figura_flicker(seed: int = 7, n_pasos: int = 1200) -> dict:
    """Cuantifica el parpadeo de periodo 2 típico de Space Invaders."""
    env = gym.make(CONFIG_FINAL["env_id"], **{**CONFIG_FINAL["make_kwargs"], "frameskip": 1})
    env.reset(seed=seed)
    frames = []
    for i in range(n_pasos):
        obs, _, term, trunc, _ = env.step(1 if i % 8 == 0 else 0)
        frames.append(obs.astype(np.int16))
        if term or trunc:
            break
    env.close()
    frames_arr = np.stack(frames)
    d1 = float(np.abs(frames_arr[1:] - frames_arr[:-1]).sum(axis=(1, 2, 3)).mean())
    d2 = float(np.abs(frames_arr[2:] - frames_arr[:-2]).sum(axis=(1, 2, 3)).mean())

    fig, eje = plt.subplots(figsize=(6, 4))
    eje.bar(["|f(t+1)-f(t)|", "|f(t+2)-f(t)|"], [d1, d2], color=["#c0392b", "#2980b9"])
    eje.set_ylabel("Diferencia absoluta media (suma de pixeles)")
    eje.set_title("Evidencia de parpadeo de periodo 2 en Space Invaders")
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "eda_flicker.png", dpi=130)
    plt.close(fig)
    return {"diff_t1": d1, "diff_t2": d2}


def figura_histograma_recompensas(distribucion: dict) -> None:
    valores = [float(k) for k in distribucion if float(k) != 0]
    conteos = [distribucion[k] for k in distribucion if float(k) != 0]
    orden = np.argsort(valores)
    fig, eje = plt.subplots(figsize=(6, 4))
    eje.bar([str(int(valores[i])) for i in orden], [conteos[i] for i in orden], color="#27ae60")
    eje.set_xlabel("Puntos otorgados en un paso (recompensa != 0)")
    eje.set_ylabel("Frecuencia (30 episodios aleatorios)")
    eje.set_title("Distribución de recompensas no nulas")
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "eda_histograma_recompensas.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Recolectando info del entorno...")
    datos = {"info_entorno": info_entorno()}

    print(f"Ejecutando {args.episodios} episodios aleatorios...")
    datos["episodios_aleatorios"] = episodios_aleatorios(args.episodios, seed=args.seed)

    print("Generando figuras de frames y parpadeo...")
    figura_frames()
    datos["flicker"] = figura_flicker()
    figura_histograma_recompensas(datos["episodios_aleatorios"]["distribucion_valores_recompensa"])

    SALIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    SALIDA_JSON.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(datos, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {SALIDA_JSON}")
    print(f"Figuras en: {FIGURAS}")


if __name__ == "__main__":
    main()
