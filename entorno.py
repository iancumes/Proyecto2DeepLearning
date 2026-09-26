"""Contrato único del entorno: mismo preprocesamiento en entrenamiento, evaluación y video.

Dos variantes conviven en este módulo, usadas por distintas iteraciones (sección 2.3
del informe):

- ``CONFIG_FINAL`` / ``envolver_final``: pila completa recomendada para Atari.
  ``gym.make(..., frameskip=1)`` + ``AtariPreprocessing(frame_skip=4, ...)`` (que
  aplica salto de frames CON max-pooling de los últimos 2 frames, recorte de
  recompensa y fin de vida como fin de episodio SOLO en entrenamiento) +
  apilado de 4 frames.
- ``CONFIG_SIMPLE`` / ``envolver_simple``: el entorno v5 "tal cual" (frameskip=4
  interno de ALE, SIN max-pooling porque ALE solo conserva el último frame de
  cada bloque) con un preprocesamiento mínimo (gris + resize + apilado), sin señal
  de vida y sin recorte de recompensa. Se usa únicamente en la iteración 2 para
  cuantificar el efecto de un preprocesamiento pobre frente al recomendado.

Ambas variantes comparten ``make_kwargs`` salvo ``frameskip``, y ambas evalúan con
recompensa cruda (sin clip) y episodios completos (3 vidas), para que el puntaje
reportado sea siempre el puntaje real del juego.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"

MAKE_KWARGS_BASE: dict[str, Any] = {
    "repeat_action_probability": 0.25,
    "full_action_space": False,
}

CONFIG_FINAL: dict[str, Any] = {
    "nombre": "final",
    "env_id": ENV_ID,
    "make_kwargs": {**MAKE_KWARGS_BASE, "frameskip": 1},
    "preprocesamiento": {
        "tipo": "atari_preprocessing",
        "frame_skip": 4,
        "noop_max": 30,
        "screen_size": 84,
        "grayscale_obs": True,
        "scale_obs": False,
        "n_stack": 4,
    },
    "entrenamiento": {
        "clip_reward": True,
        "episodic_life": True,
    },
}

CONFIG_SIMPLE: dict[str, Any] = {
    "nombre": "simple",
    "env_id": ENV_ID,
    "make_kwargs": {**MAKE_KWARGS_BASE, "frameskip": 4},
    "preprocesamiento": {
        "tipo": "resize_gris",
        "screen_size": 84,
        "n_stack": 4,
    },
    "entrenamiento": {
        "clip_reward": True,
        "episodic_life": False,
    },
}

CONFIGS_ENTORNO = {"final": CONFIG_FINAL, "simple": CONFIG_SIMPLE}


def guardar_config(cfg: dict, ruta: str | Path) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar_config(ruta: str | Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def envolver_final(env: gym.Env, cfg: dict, entrenamiento: bool) -> gym.Env:
    """Pila recomendada: AtariPreprocessing (max-pool) + vida/clip solo en train + stack."""
    pre = cfg["preprocesamiento"]
    env = AtariPreprocessing(
        env,
        frame_skip=pre["frame_skip"],
        noop_max=pre["noop_max"],
        screen_size=pre["screen_size"],
        grayscale_obs=pre["grayscale_obs"],
        terminal_on_life_loss=False,
        scale_obs=pre["scale_obs"],
    )
    entr = cfg.get("entrenamiento", {})
    if entrenamiento and entr.get("episodic_life", False):
        env = EpisodicLifeEnv(env)
    if entrenamiento and entr.get("clip_reward", False):
        env = ClipRewardEnv(env)
    env = FrameStackObservation(env, pre["n_stack"])
    return env


def envolver_simple(env: gym.Env, cfg: dict, entrenamiento: bool) -> gym.Env:
    """Entorno v5 tal cual (frameskip de ALE, sin max-pool) + gris/resize + stack."""
    pre = cfg["preprocesamiento"]
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (pre["screen_size"], pre["screen_size"]))
    entr = cfg.get("entrenamiento", {})
    if entrenamiento and entr.get("clip_reward", False):
        env = ClipRewardEnv(env)
    env = FrameStackObservation(env, pre["n_stack"])
    return env


ENVOLVER_POR_NOMBRE: dict[str, Callable[[gym.Env, dict, bool], gym.Env]] = {
    "final": envolver_final,
    "simple": envolver_simple,
}


def envolver(env: gym.Env, cfg: dict, entrenamiento: bool) -> gym.Env:
    return ENVOLVER_POR_NOMBRE[cfg["nombre"]](env, cfg, entrenamiento)


def crear_entorno_entrenamiento(
    cfg: dict, seed: int, monitor_path: str | Path | None = None
) -> gym.Env:
    env = gym.make(cfg["env_id"], **cfg["make_kwargs"])
    env = envolver(env, cfg, entrenamiento=True)
    if monitor_path is not None:
        Path(monitor_path).parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_path))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def crear_vec_entrenamiento(
    cfg: dict, n_envs: int, seed: int, run_dir: str | Path, usar_subproceso: bool = False
) -> VecEnv:
    """Construye el VecEnv de entrenamiento.

    ``usar_subproceso=True`` usa ``SubprocVecEnv`` (un proceso de SO por entorno,
    paraleliza de verdad en varios núcleos). ``False`` usa ``DummyVecEnv`` (un solo
    proceso, más simple/robusto pero sin paralelismo real); útil para depurar.
    """
    run_dir = Path(run_dir)

    def _fabrica(indice: int) -> Callable[[], gym.Env]:
        def _crear() -> gym.Env:
            # Sin extensión: Monitor añade ".monitor.csv" (formato esperado por
            # stable_baselines3.common.monitor.load_results).
            monitor_path = run_dir / "monitor" / f"env{indice}"
            return crear_entorno_entrenamiento(cfg, seed + indice, monitor_path)

        return _crear

    fabricas = [_fabrica(i) for i in range(n_envs)]
    if usar_subproceso and n_envs > 1:
        return SubprocVecEnv(fabricas)
    return DummyVecEnv(fabricas)


def crear_entorno_evaluacion(
    cfg: dict,
    video_folder: str | Path | None = None,
    name_prefix: str = "agente",
) -> gym.Env:
    """Entorno de evaluación: episodios completos, recompensa cruda, sin clip.

    Si se da ``video_folder``, usa ``agentes_ale.crear_entorno`` (Laboratorio 5)
    para grabar en la capa base con ``RecordVideo`` antes de aplicar el
    preprocesamiento del agente.
    """
    from agentes_ale import crear_entorno as crear_entorno_lab5

    if video_folder is not None:
        env = crear_entorno_lab5(
            cfg["env_id"],
            video_folder=video_folder,
            name_prefix=name_prefix,
            **cfg["make_kwargs"],
        )
    else:
        env = gym.make(cfg["env_id"], **cfg["make_kwargs"])
    return envolver(env, cfg, entrenamiento=False)
