"""Entrena una iteración (PPO o DQN) según un config de configs/*.json.

Guarda, en resultados/<run>/: checkpoints/, monitor/ (recompensa cruda por episodio
de cada entorno), eval/ (evaluaciones periódicas greedy), tb/ (TensorBoard),
valores.csv (monitor de divergencia V/Q sobre estados fijos), final.zip y
final.json (contrato de entorno + hiperparámetros + pasos + versiones).

Uso:
    python scripts/entrenar.py --config configs/iter3_ppo_final.json --threads 8 --hasta 22:05
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from entorno import (  # noqa: E402
    CONFIGS_ENTORNO,
    crear_entorno_evaluacion,
    crear_vec_entrenamiento,
    guardar_config,
)
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import get_linear_fn  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def cargar_config_entrenamiento(ruta_json: str | Path) -> dict:
    return json.loads(Path(ruta_json).read_text(encoding="utf-8"))


class LimiteDeReloj(BaseCallback):
    """Detiene el entrenamiento (con guardado limpio) al llegar a una hora HH:MM de hoy."""

    def __init__(self, hasta: str | None, verbose: int = 0):
        super().__init__(verbose)
        self.limite_ts: float | None = None
        if hasta:
            hoy = dt.date.today()
            hh, mm = (int(x) for x in hasta.split(":"))
            self.limite_ts = dt.datetime.combine(hoy, dt.time(hh, mm)).timestamp()

    def _on_step(self) -> bool:
        if self.limite_ts is not None and time.time() >= self.limite_ts:
            print(f"[entrenar] limite de reloj alcanzado en el paso {self.num_timesteps}", flush=True)
            return False
        return True


class Bitacora(BaseCallback):
    """Progreso legible en stdout, independiente del logger interno de SB3."""

    def __init__(self, cada: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.cada = cada
        self._ultimo = 0
        self._inicio = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._ultimo >= self.cada:
            self._ultimo = self.num_timesteps
            buffer = self.model.ep_info_buffer
            media = float(np.mean([e["r"] for e in buffer])) if buffer else float("nan")
            transcurrido = (time.time() - self._inicio) / 60
            print(
                f"[entrenar] pasos={self.num_timesteps} "
                f"recompensa_media_100ep={media:.1f} "
                f"transcurrido_min={transcurrido:.1f}",
                flush=True,
            )
        return True


class MonitorDivergencia(BaseCallback):
    """Registra el valor medio V(s) (PPO) o max_a Q(s,a) (DQN) sobre estados fijos.

    Permite detectar divergencia o estancamiento (sección 2.3 del informe) sin
    afectar el entrenamiento: cualquier error aquí se ignora tras avisar una vez.
    """

    def __init__(self, estados_fijos: np.ndarray, ruta_csv: Path, cada_pasos: int, algoritmo: str, verbose: int = 0):
        super().__init__(verbose)
        self.estados_fijos = estados_fijos
        self.ruta_csv = Path(ruta_csv)
        self.cada_pasos = cada_pasos
        self.algoritmo = algoritmo
        self._ultimo = 0
        self._deshabilitado = False
        self.ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_csv.write_text("pasos,valor_medio\n", encoding="utf-8")

    def _on_step(self) -> bool:
        if self._deshabilitado:
            return True
        if self.num_timesteps - self._ultimo < self.cada_pasos:
            return True
        self._ultimo = self.num_timesteps
        try:
            obs_tensor, _ = self.model.policy.obs_to_tensor(self.estados_fijos)
            with torch.no_grad():
                if self.algoritmo == "ppo":
                    valores = self.model.policy.predict_values(obs_tensor)
                else:
                    valores = self.model.q_net(obs_tensor).max(dim=1).values
                valor_medio = float(valores.mean().item())
            with self.ruta_csv.open("a", encoding="utf-8") as f:
                f.write(f"{self.num_timesteps},{valor_medio}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[entrenar] aviso: monitor de divergencia deshabilitado ({exc})", flush=True)
            self._deshabilitado = True
        return True


def recolectar_estados_fijos(cfg_entorno: dict, n: int = 500, seed: int = 999) -> np.ndarray:
    env = crear_entorno_evaluacion(cfg_entorno)
    estados = []
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    while len(estados) < n:
        estados.append(np.array(obs))
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.stack(estados[:n])


def construir_modelo(cfg_entrenamiento: dict, vec_env, tb_log: Path, seed: int):
    algoritmo = cfg_entrenamiento["algoritmo"]
    hp = dict(cfg_entrenamiento["hiperparametros"])
    policy = cfg_entrenamiento.get("policy", "CnnPolicy")
    if algoritmo == "ppo":
        lr0 = hp.pop("learning_rate_inicial")
        clip0 = hp.pop("clip_range_inicial")
        return PPO(
            policy,
            vec_env,
            seed=seed,
            verbose=1,
            device="cpu",
            tensorboard_log=str(tb_log),
            learning_rate=get_linear_fn(lr0, 0.0, 1.0),
            clip_range=get_linear_fn(clip0, 0.0, 1.0),
            **hp,
        )
    if algoritmo == "dqn":
        return DQN(
            policy,
            vec_env,
            seed=seed,
            verbose=1,
            device="cpu",
            tensorboard_log=str(tb_log),
            **hp,
        )
    raise ValueError(f"Algoritmo no soportado: {algoritmo!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", default=None, help="Carpeta en resultados/ (por defecto, el id del config)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--hasta", default=None, help="Hora limite HH:MM de hoy")
    parser.add_argument("--seed", type=int, default=23236)
    parser.add_argument("--continuar", default=None, help="Ruta a un final.zip para reanudar")
    parser.add_argument("--checkpoint-cada", type=int, default=100_000)
    parser.add_argument("--eval-cada", type=int, default=250_000)
    parser.add_argument(
        "--vec", choices=["subproc", "dummy"], default="subproc",
        help="subproc = un proceso de SO por entorno (paralelismo real); dummy = un solo proceso",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    cfg_entrenamiento = cargar_config_entrenamiento(args.config)
    cfg_entorno = CONFIGS_ENTORNO[cfg_entrenamiento["entorno"]]
    n_envs = cfg_entrenamiento["n_envs"]
    nombre_run = args.run or cfg_entrenamiento["id"]
    run_dir = RAIZ / "resultados" / nombre_run
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[entrenar] run={nombre_run} algoritmo={cfg_entrenamiento['algoritmo']} entorno={cfg_entorno['nombre']} n_envs={n_envs}", flush=True)

    vec_env = crear_vec_entrenamiento(
        cfg_entorno, n_envs, args.seed, run_dir, usar_subproceso=(args.vec == "subproc")
    )
    eval_env = DummyVecEnv([lambda: crear_entorno_evaluacion(cfg_entorno)])

    if args.continuar:
        Algoritmo = PPO if cfg_entrenamiento["algoritmo"] == "ppo" else DQN
        modelo = Algoritmo.load(args.continuar, env=vec_env, device="cpu", tensorboard_log=str(run_dir / "tb"))
        reset_num_timesteps = False
        print(f"[entrenar] reanudado desde {args.continuar} en el paso {modelo.num_timesteps}", flush=True)
    else:
        modelo = construir_modelo(cfg_entrenamiento, vec_env, run_dir / "tb", args.seed)
        reset_num_timesteps = True

    callbacks: list[BaseCallback] = [
        LimiteDeReloj(args.hasta),
        Bitacora(cada=20_000),
        CheckpointCallback(
            save_freq=max(args.checkpoint_cada // n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=nombre_run,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "eval"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(args.eval_cada // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]
    try:
        estados_fijos = recolectar_estados_fijos(cfg_entorno)
        callbacks.append(
            MonitorDivergencia(
                estados_fijos, run_dir / "valores.csv", cada_pasos=10_000, algoritmo=cfg_entrenamiento["algoritmo"]
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[entrenar] aviso: no se pudo iniciar el monitor de divergencia: {exc}", flush=True)

    inicio = time.time()
    try:
        modelo.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="tb",
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    finally:
        duracion = time.time() - inicio
        ruta_final = run_dir / "final.zip"
        modelo.save(str(ruta_final))
        metadatos = {
            "id": nombre_run,
            "algoritmo": cfg_entrenamiento["algoritmo"],
            "entorno": cfg_entorno,
            "hiperparametros": cfg_entrenamiento["hiperparametros"],
            "n_envs": n_envs,
            "pasos_alcanzados": int(modelo.num_timesteps),
            "duracion_segundos": duracion,
            "seed": args.seed,
            "versiones": {
                "python": platform.python_version(),
                "stable_baselines3": __import__("stable_baselines3").__version__,
                "torch": torch.__version__,
                "gymnasium": __import__("gymnasium").__version__,
            },
        }
        guardar_config(metadatos, run_dir / "final.json")
        print(
            f"[entrenar] listo: {modelo.num_timesteps} pasos en {duracion / 60:.1f} min -> {ruta_final}",
            flush=True,
        )

        registro = RAIZ / "resultados" / "registro_iteraciones.csv"
        es_nuevo = not registro.exists()
        with registro.open("a", encoding="utf-8") as f:
            if es_nuevo:
                f.write("id,algoritmo,entorno,n_envs,pasos,duracion_min\n")
            f.write(
                f"{nombre_run},{cfg_entrenamiento['algoritmo']},{cfg_entorno['nombre']},"
                f"{n_envs},{modelo.num_timesteps},{duracion / 60:.1f}\n"
            )

        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
