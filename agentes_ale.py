"""Utilidades reutilizables para agentes de Gymnasium y Arcade Learning Environment.

Base tomada del Laboratorio 5 (crear_entorno, agente_aleatorio, agente_regla_simple,
ejecutar_episodio, generar_video_agente). Única adición para el Proyecto 2:
``generar_video_agente`` acepta un parámetro opcional ``envolver`` para aplicar el
preprocesamiento (AtariPreprocessing + FrameStack) al entorno que graba, de modo que
el agente entrenado reciba observaciones idénticas a las de entrenamiento/evaluación,
mientras que el video conserva los fotogramas RGB originales (la grabación ocurre en
la capa base, antes de aplicar ``envolver``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import gymnasium as gym


FuncionAgente = Callable[[Any, gym.Env], Any]


def _registrar_ale() -> None:
    """Registra los entornos ALE si ale-py está instalado."""
    try:
        import ale_py
    except ImportError:
        return
    gym.register_envs(ale_py)


def crear_entorno(
    nombre_entorno: str,
    video_folder: str | Path | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
    name_prefix: str = "agente",
    **kwargs: Any,
) -> gym.Env:
    """Crea un entorno Gymnasium y, opcionalmente, graba episodios en MP4.

    Los argumentos adicionales se reenvían a ``gym.make``. Si se solicita video,
    el entorno usa ``render_mode='rgb_array'`` y graba todos los episodios salvo
    que se proporcione otro ``episode_trigger``.
    """
    if not nombre_entorno or not isinstance(nombre_entorno, str):
        raise ValueError("nombre_entorno debe ser una cadena no vacía")
    if not name_prefix:
        raise ValueError("name_prefix no puede estar vacío")

    _registrar_ale()
    parametros = dict(kwargs)
    if video_folder is not None:
        render_mode = parametros.get("render_mode")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(
                "RecordVideo requiere render_mode='rgb_array'; "
                f"se recibió {render_mode!r}"
            )
        parametros["render_mode"] = "rgb_array"

    try:
        env = gym.make(nombre_entorno, **parametros)
    except gym.error.DependencyNotInstalled as exc:
        raise RuntimeError(
            f"No se pudo crear {nombre_entorno!r}: falta una dependencia de renderizado."
        ) from exc

    if video_folder is None:
        return env

    carpeta = Path(video_folder)
    carpeta.parent.mkdir(parents=True, exist_ok=True)
    trigger = episode_trigger if episode_trigger is not None else (lambda _: True)
    return gym.wrappers.RecordVideo(
        env,
        video_folder=str(carpeta),
        episode_trigger=trigger,
        name_prefix=name_prefix,
        disable_logger=True,
    )


def agente_aleatorio(observation: Any, env: gym.Env) -> Any:
    """Devuelve una única acción aleatoria del espacio de acciones del entorno."""
    del observation
    return env.action_space.sample()


def agente_regla_simple(observation: Any, env: gym.Env) -> int:
    """Dispara continuamente en un entorno Atari que incluya la acción FIRE.

    Es una regla demostrativa específica de Atari y no garantiza un rendimiento
    mayor que el agente aleatorio.
    """
    del observation
    obtener_acciones = getattr(env.unwrapped, "get_action_meanings", None)
    if obtener_acciones is None:
        raise ValueError("El entorno no expone significados de acciones de Atari")
    significados = list(obtener_acciones())
    try:
        return significados.index("FIRE")
    except ValueError as exc:
        raise ValueError("El entorno no incluye la acción FIRE") from exc


def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: FuncionAgente,
    max_steps: int | None = 10_000,
    *,
    seed: int | None = None,
) -> dict[str, int | float | bool]:
    """Ejecuta un episodio y devuelve pasos, retorno y causa de finalización.

    ``pasos`` cuenta llamadas a ``env.step``; en Atari cada llamada puede avanzar
    varios frames debido a ``frameskip``. ``max_steps=None`` elimina el límite
    local y deja que el entorno determine el final.
    """
    if max_steps is not None and (isinstance(max_steps, bool) or max_steps <= 0):
        raise ValueError("max_steps debe ser un entero positivo o None")
    if not callable(funcion_agente):
        raise TypeError("funcion_agente debe ser invocable")

    if seed is not None:
        env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    pasos = 0
    recompensa_total = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if max_steps is not None and pasos >= max_steps:
            break
        accion = funcion_agente(observation, env)
        observation, recompensa, terminated, truncated, _ = env.step(accion)
        recompensa_total += float(recompensa)
        pasos += 1

    limite_alcanzado = bool(
        max_steps is not None
        and pasos >= max_steps
        and not (terminated or truncated)
    )
    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "limite_alcanzado": limite_alcanzado,
    }


def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: FuncionAgente,
    video_folder: str | Path,
    name_prefix: str,
    n_episodios: int = 1,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    envolver: Callable[[gym.Env], gym.Env] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Graba episodios en una subcarpeta única y asocia videos con métricas.

    Las semillas son consecutivas a partir de ``seed``. El entorno siempre se
    cierra, incluso si el agente o el codificador de video producen un error.

    ``envolver``, si se proporciona, se aplica DESPUÉS de crear el entorno base
    (que ya incluye ``RecordVideo``), para dar al agente las mismas observaciones
    preprocesadas que vio durante el entrenamiento sin alterar el video grabado
    (la grabación captura los fotogramas RGB de la capa base).
    """
    if isinstance(n_episodios, bool) or not isinstance(n_episodios, int) or n_episodios <= 0:
        raise ValueError("n_episodios debe ser un entero positivo")
    if not name_prefix:
        raise ValueError("name_prefix no puede estar vacío")

    carpeta_ejecucion = Path(video_folder) / f"{name_prefix}_{uuid4().hex[:10]}"
    env = crear_entorno(
        nombre_entorno,
        video_folder=carpeta_ejecucion,
        episode_trigger=lambda _: True,
        name_prefix=name_prefix,
        **kwargs,
    )
    if envolver is not None:
        env = envolver(env)
    metricas: list[dict[str, Any]] = []
    try:
        for episodio in range(n_episodios):
            semilla = None if seed is None else seed + episodio
            resultado = ejecutar_episodio(
                env, funcion_agente, max_steps=max_steps, seed=semilla
            )
            metricas.append(
                {"episodio": episodio + 1, "semilla": semilla, **resultado}
            )
    finally:
        env.close()

    def indice_episodio(ruta: Path) -> int:
        coincidencia = re.search(r"-episode-(\d+)$", ruta.stem)
        if coincidencia is None:
            raise RuntimeError(f"Nombre de video inesperado: {ruta.name}")
        return int(coincidencia.group(1))

    videos = sorted(carpeta_ejecucion.glob("*.mp4"), key=indice_episodio)
    if len(videos) != n_episodios:
        raise RuntimeError(
            f"Se esperaban {n_episodios} videos y se generaron {len(videos)} "
            f"en {carpeta_ejecucion}"
        )
    for registro, video in zip(metricas, videos, strict=True):
        registro["video"] = str(video.resolve())
    return metricas
