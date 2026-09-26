from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from agentes_ale import (
    agente_aleatorio,
    agente_regla_simple,
    crear_entorno,
    ejecutar_episodio,
    generar_video_agente,
)


class EntornoControlado(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, termina_en=3, trunca=False, render_mode=None):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        self.termina_en = termina_en
        self.trunca = trunca
        self.render_mode = render_mode
        self.pasos = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pasos = 0
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        self.pasos += 1
        terminado = self.pasos >= self.termina_en and not self.trunca
        truncado = self.pasos >= self.termina_en and self.trunca
        return np.zeros((8, 8, 3), dtype=np.uint8), 1.5, terminado, truncado, {}

    def render(self):
        return np.full((8, 8, 3), self.pasos, dtype=np.uint8)


def test_agente_aleatorio_muestrea_una_vez(monkeypatch):
    env = EntornoControlado()
    llamadas = []
    monkeypatch.setattr(env.action_space, "sample", lambda: llamadas.append(1) or 1)
    assert agente_aleatorio(None, env) == 1
    assert llamadas == [1]


def test_episodio_terminado_acumula_recompensa():
    env = EntornoControlado(termina_en=3)
    resultado = ejecutar_episodio(env, lambda obs, entorno: 0, seed=7)
    assert resultado == {
        "pasos": 3,
        "recompensa_total": 4.5,
        "terminated": True,
        "truncated": False,
        "limite_alcanzado": False,
    }


def test_episodio_truncado():
    resultado = ejecutar_episodio(
        EntornoControlado(termina_en=2, trunca=True), lambda obs, entorno: 0
    )
    assert resultado["pasos"] == 2
    assert resultado["truncated"] is True
    assert resultado["terminated"] is False


def test_limite_local_y_validacion():
    resultado = ejecutar_episodio(
        EntornoControlado(termina_en=99), lambda obs, entorno: 0, max_steps=2
    )
    assert resultado["pasos"] == 2
    assert resultado["limite_alcanzado"] is True
    with pytest.raises(ValueError):
        ejecutar_episodio(EntornoControlado(), lambda obs, entorno: 0, max_steps=0)


def test_regla_simple_resuelve_fire():
    env = EntornoControlado()
    env.get_action_meanings = lambda: ["NOOP", "LEFT", "FIRE"]
    assert agente_regla_simple(None, env) == 2


def test_crear_entorno_generico_y_render_incompatible(tmp_path):
    env = crear_entorno("CartPole-v1")
    try:
        assert env.action_space.n == 2
    finally:
        env.close()
    with pytest.raises(ValueError):
        crear_entorno("CartPole-v1", tmp_path, render_mode="human")


def test_generar_video_asocia_metricas(tmp_path: Path):
    env_id = "Lab5/EntornoControlado-v0"
    if env_id not in gym.registry:
        gym.register(env_id, entry_point=EntornoControlado)
    resultados = generar_video_agente(
        env_id,
        lambda obs, entorno: 0,
        tmp_path,
        "prueba",
        n_episodios=2,
        seed=10,
        termina_en=2,
    )
    assert [r["semilla"] for r in resultados] == [10, 11]
    assert all(r["pasos"] == 2 and r["terminated"] for r in resultados)
    assert all(Path(r["video"]).is_file() for r in resultados)
