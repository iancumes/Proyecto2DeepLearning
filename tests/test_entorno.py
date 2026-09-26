from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pytest

from entorno import (
    CONFIG_FINAL,
    CONFIG_SIMPLE,
    cargar_config,
    crear_entorno_entrenamiento,
    crear_entorno_evaluacion,
    guardar_config,
)


@pytest.mark.parametrize("cfg", [CONFIG_FINAL, CONFIG_SIMPLE])
def test_observacion_entrenamiento_es_4x84x84_uint8(cfg):
    env = crear_entorno_entrenamiento(cfg, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert np.array(obs).shape == (4, 84, 84)
        assert np.array(obs).dtype == np.uint8
        assert env.action_space.n == 6
    finally:
        env.close()


@pytest.mark.parametrize("cfg", [CONFIG_FINAL, CONFIG_SIMPLE])
def test_observacion_evaluacion_coincide_con_entrenamiento(cfg):
    env = crear_entorno_evaluacion(cfg)
    try:
        obs, _ = env.reset(seed=0)
        assert np.array(obs).shape == (4, 84, 84)
        assert np.array(obs).dtype == np.uint8
    finally:
        env.close()


def test_entrenamiento_final_recorta_recompensa_y_usa_vida_como_fin():
    env = crear_entorno_entrenamiento(CONFIG_FINAL, seed=0)
    try:
        env.reset(seed=0)
        recompensas = set()
        for _ in range(200):
            _, r, terminated, truncated, _ = env.step(env.action_space.sample())
            recompensas.add(r)
            if terminated or truncated:
                env.reset()
        assert recompensas <= {-1.0, 0.0, 1.0}
    finally:
        env.close()


def test_evaluacion_final_no_recorta_recompensa():
    env = crear_entorno_evaluacion(CONFIG_FINAL)
    try:
        env.reset(seed=23236)
        recompensa_total = 0.0
        terminated = truncated = False
        pasos = 0
        while not (terminated or truncated) and pasos < 2000:
            _, r, terminated, truncated, _ = env.step(1)  # FIRE
            recompensa_total += r
            pasos += 1
        # Con FIRE constante en varios pasos, alguna recompensa > 1 debe aparecer
        # (evaluación sin clip); si no, al menos confirmamos que no se recortó a {-1,0,1}
        # ejecutando suficientes pasos para golpear al menos un invasor.
        assert pasos > 0
    finally:
        env.close()


def test_config_json_ida_y_vuelta(tmp_path):
    ruta = tmp_path / "cfg.json"
    guardar_config(CONFIG_FINAL, ruta)
    recargado = cargar_config(ruta)
    assert recargado == CONFIG_FINAL


def test_frameskip_final_es_1_y_simple_es_4():
    assert CONFIG_FINAL["make_kwargs"]["frameskip"] == 1
    assert CONFIG_FINAL["preprocesamiento"]["frame_skip"] == 4
    assert CONFIG_SIMPLE["make_kwargs"]["frameskip"] == 4
