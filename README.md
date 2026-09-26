# Proyecto2DeepLearning

Proyecto 2 de CC3092 - Deep Learning y Sistemas Inteligentes: agente de Reinforcement
Learning (PPO, Stable-Baselines3) entrenado para jugar `ALE/SpaceInvaders-v5`
(Gymnasium + ale-py), con una iteración comparativa en DQN y otra en PPO sin el
preprocesamiento recomendado.

**Autor:** Ian Cumes - carné 23236

Ver [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md) para el plan de trabajo completo
(cronograma, decisiones técnicas y justificación de cada iteración).

## Entregables

- `entregables/Proyecto2_Informe_Ian_Cumes_23236.pdf` (+ `.docx` editable): trabajo escrito completo (secciones 2.1-2.6).
- `entregables/space_invaders_ppo_final.mp4`: episodio completo del modelo final (PPO, política greedy).
- `entregables/metricas_video.json`: puntajes de los 5 episodios de evaluación, sha256 y metadatos del video.
- `entregables/figuras/`: figuras del EDA y de las curvas de entrenamiento/evaluación de cada iteración.
- `modelos/ppo_space_invaders_final.zip` (+ `.json`): pesos del modelo entregado y su contrato de entorno (preprocesamiento exacto).
- `resultados/`: bitácora de iteraciones, logs de TensorBoard, checkpoints y evaluaciones periódicas de cada corrida.

## Estructura del código

- `agentes_ale.py`: módulo del Laboratorio 5 (`crear_entorno`, `agente_aleatorio`, `agente_regla_simple`, `ejecutar_episodio`, `generar_video_agente`), con una única adición retrocompatible: `generar_video_agente` acepta `envolver` para aplicar el preprocesamiento del agente entrenado sin alterar el video grabado.
- `entorno.py`: contrato único de preprocesamiento (idéntico en entrenamiento y evaluación). Dos variantes: `CONFIG_FINAL` (recomendada: max-pooling + fin de vida + recorte de recompensa solo en entrenamiento) y `CONFIG_SIMPLE` (entorno v5 tal cual, usada solo en la iteración 2 para comparar).
- `agente.py`: carga un modelo entrenado (`cargar_modelo`) y lo adapta a la firma de función de agente del Laboratorio 5 (`politica_greedy`).
- `configs/*.json`: hiperparámetros y variante de entorno de cada iteración.
- `scripts/entrenar.py`: entrena una iteración (PPO o DQN), con checkpoints, evaluación periódica greedy y un monitor de divergencia de V(s)/Q(s,a).
- `scripts/evaluar.py`: **el script del día de la presentación** — carga un modelo y evalúa 5 episodios greedy, reporta media/desviación/máximo.
- `scripts/generar_video.py`: graba el episodio entregable con el mismo protocolo de evaluación.
- `scripts/analizar_entorno.py`: EDA del entorno (sección 2.1).
- `scripts/graficar.py`: figuras de entrenamiento/evaluación/diagnóstico por iteración.
- `scripts/generar_informe.py`: genera el informe (DOCX + PDF) a partir de todos los resultados anteriores.

## Instalación (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

ALE 0.12.1 incluye las ROMs soportadas; no se descarga ni versiona una ROM aparte.

## Cargar el modelo final y evaluarlo (día de la presentación)

```powershell
.\.venv\Scripts\python.exe scripts\evaluar.py --modelo modelos\ppo_space_invaders_final.zip --episodios 5 --seed 23236
```

Esto imprime el puntaje de cada episodio y la media/desviación/máximo (la métrica
de la competencia), y guarda el detalle en `resultados/evaluaciones/`. Para
regenerar el video oficial con el mismo protocolo:

```powershell
.\.venv\Scripts\python.exe scripts\generar_video.py --modelo modelos\ppo_space_invaders_final.zip --seed 23236
```

## Reproducir el entrenamiento de cada iteración

```powershell
# Iteración 3 (PPO, preprocesamiento final) - modelo entregado
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter3_ppo_final.json --threads 4 --hasta 22:05

# Iteración 1 (DQN, comparación de algoritmo)
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter1_dqn.json --threads 3 --hasta 21:15 --vec dummy

# Iteración 2 (PPO sobre el entorno v5 "tal cual", sin max-pool ni señal de vida)
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter2_ppo_v5_directo.json --threads 4 --hasta 21:30

# Reanudar una iteración desde su último checkpoint
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter3_ppo_final.json --continuar resultados\iter3_ppo_final\final.zip --timesteps 8000000

# Ver las curvas en vivo
.\.venv\Scripts\tensorboard.exe --logdir resultados

# Regenerar el EDA, las figuras y el informe
.\.venv\Scripts\python.exe scripts\analizar_entorno.py --episodios 30
.\.venv\Scripts\python.exe scripts\graficar.py
.\.venv\Scripts\python.exe scripts\generar_informe.py

# Pruebas
.\.venv\Scripts\python.exe -m pytest
```

`--vec subproc` (por defecto) usa un proceso de sistema operativo por entorno
paralelo (paralelismo real en varios núcleos); `--vec dummy` usa un solo proceso
(más simple, útil con `n_envs=1` o para depurar).

## Preprocesamiento (idéntico en entrenamiento y evaluación)

```text
gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.25, full_action_space=False)
 └─ AtariPreprocessing(frame_skip=4, noop_max=30, screen_size=84, grayscale_obs=True)  # max-pool de los últimos 2 frames
 └─ EpisodicLifeEnv + ClipRewardEnv     ← SOLO en entrenamiento
 └─ FrameStackObservation(4)            → observación (4, 84, 84) uint8
```

La evaluación (y el video) usan la partida completa (3 vidas) con recompensa
**cruda** (sin recortar) y política greedy (`deterministic=True`). El contrato
completo se guarda en el JSON junto a cada modelo (`modelos/*.json`,
`resultados/<iter>/final.json`) para que cargarlo reconstruya exactamente el
mismo preprocesamiento sin pasos manuales.

## Solución de problemas

- **Entorno ALE no registrado:** confirme que `ale-py==0.12.1` está instalado; `entorno.py` registra los entornos automáticamente al importarse.
- **`SubprocVecEnv` falla en Windows:** use `--vec dummy` (un solo proceso, más lento pero sin multiprocessing).
- **No se crea el MP4:** reinstale `imageio-ffmpeg==0.6.0` y verifique permisos de escritura en `entregables/`.
- **El informe no exporta a PDF:** requiere Microsoft Word instalado (usa `pywin32` para exportar); si no está disponible, abra el `.docx` y use "Guardar como PDF" manualmente.

## Fuentes principales

- Mnih et al. (2015). *Human-level control through deep reinforcement learning* (DQN). Nature.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Machado et al. (2018). *Revisiting the Arcade Learning Environment*. JAIR.
- van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
- Wang et al. (2016). *Dueling Network Architectures for Deep RL*. ICML.
- Huang et al. (2022). *The 37 Implementation Details of Proximal Policy Optimization*. ICLR Blog Track.
- Raffin et al. (2021). *Stable-Baselines3: Reliable RL Implementations*. JMLR.
- [Arcade Learning Environment](https://ale.farama.org/) · [Space Invaders en ALE](https://ale.farama.org/environments/space_invaders/) · [Wrappers de Gymnasium](https://gymnasium.farama.org/api/wrappers/)
