"""Carga del modelo entrenado y adaptación al contrato de función de agente del Lab 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from entorno import cargar_config


def cargar_modelo(ruta_zip: str | Path) -> tuple[Any, dict]:
    """Carga los pesos y el contrato de entorno (JSON hermano) de un modelo entrenado.

    Devuelve ``(modelo, cfg_entorno)``. ``cfg_entorno`` es el diccionario de
    ``entorno.py`` (CONFIG_FINAL o CONFIG_SIMPLE) usado durante el entrenamiento,
    para que la evaluación reconstruya exactamente el mismo preprocesamiento.
    """
    ruta_zip = Path(ruta_zip)
    ruta_json = ruta_zip.with_suffix(".json")
    if not ruta_json.is_file():
        raise FileNotFoundError(
            f"No se encontró el contrato de entorno junto al modelo: {ruta_json}"
        )
    metadatos = cargar_config(ruta_json)
    algoritmo = metadatos["algoritmo"].lower()

    if algoritmo == "ppo":
        from stable_baselines3 import PPO as Algoritmo
    elif algoritmo == "dqn":
        from stable_baselines3 import DQN as Algoritmo
    else:
        raise ValueError(f"Algoritmo no soportado: {algoritmo!r}")

    modelo = Algoritmo.load(str(ruta_zip), device="cpu")
    return modelo, metadatos["entorno"]


def politica_greedy(modelo: Any):
    """Función de agente (firma ``(observation, env) -> accion``) con política greedy."""

    def _funcion_agente(observation: Any, env: gym.Env) -> int:
        del env
        accion, _ = modelo.predict(observation, deterministic=True)
        return int(accion)

    return _funcion_agente
