# Plan de trabajo — Proyecto 2: agente de RL para Space Invaders

**Autor:** Ian Cumes · 23236 · CC3092 Deep Learning y Sistemas Inteligentes
**Hoy:** viernes 25/09/2026 · **Meta:** todo entregado y en GitHub antes de las **23:00**
**Repo:** https://github.com/iancumes/Proyecto2DeepLearning (público)

---

## 0. Punto de partida (verificado hoy a las 19:15)

| Tema | Estado real | Consecuencia para el plan |
|---|---|---|
| Laboratorio 5 | **No entrenó ningún modelo.** Solo tiene la infraestructura (`crear_entorno`, `ejecutar_episodio`, `generar_video_agente`) y un agente aleatorio: 365 pts en 572 decisiones (semilla 23236). | El entrenamiento empieza de cero hoy. Se reutilizan `agentes_ale.py`, sus tests y el manifiesto JSON del video. |
| Repo Proyecto 2 | Vacío (solo README). Remoto OK; `gh` autenticado como `iancumes`. | Push directo a `main`. |
| Hardware | i9-13900H (14 núcleos / 20 hilos), 15.6 GB de RAM (**solo 1.6 GB libres ahora**), **sin GPU NVIDIA**, conectado a corriente. | Entrenamiento **solo en CPU**: ≈ 2 h de cómputo ≈ 1.5–2.5 M pasos (estimado; se mide en el smoke test). Hay que liberar RAM. |
| Velocidad del emulador | 3,544 pasos/s en crudo y 2,336 pasos/s con preprocesamiento (1 proceso). | El cuello de botella es la red en CPU, no ALE → basta `DummyVecEnv` (sin subprocesos). |
| Entorno | `Discrete(6)`: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE · 3 vidas · sticky actions 0.25 · frameskip 4 · tope de 108,000 frames. En 4,000 pasos aleatorios, **98 % de los pasos dan recompensa 0**; valores vistos: 5, 10, 15, 20, 25 y 30. | Recompensa dispersa y heterogénea → clipping en entrenamiento y puntaje crudo en evaluación. |
| Dependencias | SB3 2.9.0 pide `gymnasium<2.0`, `torch>=2.8` y Python ≥ 3.10. torch 2.14 está disponible. | Compatible con gymnasium 1.3.0, ale-py 0.12.1 y Python 3.13 (igual que el Lab 5). |

---

## 1. Decisiones técnicas

1. **Algoritmo final: PPO** (Stable-Baselines3) con 8 entornos en paralelo. En CPU el costo dominante es la red y PPO reparte cada actualización sobre 1,024 transiciones de 8 partidas. Además, no necesita replay buffer (hay poca RAM) y es estable con los hiperparámetros estándar de Atari. **DQN** se entrena como iteración comparativa para justificar la elección con evidencia (lo pide la sección 2.2).
2. **Librería:** SB3, permitido por el enunciado (2.2). Se documenta cada decisión; no hay tiempo para depurar una implementación propia.
3. **Contrato del entorno.** Es idéntico en entrenamiento, evaluación y video, vive en un solo archivo (`entorno.py`) y se guarda como JSON junto a los pesos:

   ```text
   gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.25, full_action_space=False)
    └─ [RecordVideo]                  ← solo al grabar (crear_entorno del Lab 5)
    └─ AtariPreprocessing(frame_skip=4, noop_max=30, screen_size=84, grayscale_obs=True,
                          terminal_on_life_loss=False, scale_obs=False)   ← max-pool de los 2 últimos frames
    └─ Monitor                        ← puntaje crudo por partida completa (solo entrenamiento)
    └─ EpisodicLifeEnv + ClipRewardEnv ← SOLO entrenamiento
    └─ FrameStackObservation(4)       → obs (4, 84, 84) uint8
   ```

   - Con `frameskip=1` en `gym.make` y `frame_skip=4` en el wrapper se mantienen las 4 repeticiones por decisión del v5 por defecto, pero con max-pooling de los 2 últimos frames. Space Invaders dibuja los proyectiles en frames alternos (lo documenta Mnih et al. 2013 y se verifica en el EDA); sin max-pool el agente puede no ver las balas. **Nunca** se combinan el frameskip del v5 y el del wrapper, porque serían 16 frames por decisión; `AtariPreprocessing` lanza un error si pasa.
   - Se conservan las sticky actions de 0.25: se entrena con la misma estocasticidad con la que se evalúa.
   - La evaluación usa la partida completa (3 vidas), recompensa **sin** recortar y política greedy (`deterministic=True`).
4. **Protocolo de evaluación (igual al de la competencia):** 5 episodios con semillas 23236–23240. Se reportan media ± desviación y **máximo**; el ranking usa el máximo.

---

## 2. Estructura del repositorio

```text
Proyecto2DeepLearning/
├── README.md                    # reproducir, cargar pesos, evaluar el día de la presentación
├── requirements.txt             # versiones fijadas
├── .gitignore / .gitattributes / pytest.ini   # del Lab 5, ampliados
├── agentes_ale.py               # módulo del Lab 5 (+ parámetro `envolver` en generar_video_agente)
├── entorno.py                   # contrato del entorno: entrenamiento / evaluación / vec
├── agente.py                    # cargar_modelo(), politica_greedy() → función de agente estilo Lab 5
├── configs/
│   ├── iter1_dqn.json
│   ├── iter2_ppo_v5_directo.json
│   └── iter3_ppo_final.json
├── scripts/
│   ├── analizar_entorno.py      # EDA → figuras + resultados/eda.json
│   ├── entrenar.py              # --config --threads --timesteps --hasta HH:MM --continuar
│   ├── evaluar.py               # 5 episodios greedy → tabla + JSON (script de la presentación)
│   ├── generar_video.py         # episodio completo .mp4 + manifiesto (sha256, frames, puntaje)
│   ├── graficar.py              # curvas por iteración → entregables/figuras/
│   └── generar_informe.py       # informe/informe.md + figuras → DOCX → PDF (vía Word)
├── informe/informe.md           # texto del trabajo escrito (se redacta mientras entrena)
├── modelos/
│   ├── ppo_space_invaders_final.zip
│   └── ppo_space_invaders_final.json   # contrato + hiperparámetros + pasos + versiones
├── resultados/<iteracion>/      # config.json, progress.csv, monitor/*.csv, eval/evaluations.npz
│                                # (checkpoints/ y tb/ van en .gitignore)
├── resultados/evaluaciones/*.json
├── resultados/registro_iteraciones.csv   # bitácora (recomendación 5 del enunciado)
├── entregables/
│   ├── Proyecto2_Informe_Ian_Cumes_23236.pdf / .docx
│   ├── space_invaders_ppo_final.mp4
│   ├── metricas_video.json
│   └── figuras/*.png
└── tests/                       # tests del Lab 5 + test_entorno.py
```

---

## 3. Iteraciones planificadas (sección 2.3 del informe)

Las tres corridas se lanzan en paralelo alrededor de las 20:00, cada una con límite de reloj. Todas guardan checkpoints, así que ninguna se pierde si se corta.

| Iter | Cambio respecto a la anterior | Pasos (máx.) | Hilos | Corta a las | Qué demuestra |
|---|---|---|---|---|---|
| 0 | Agente aleatorio del Lab 5, sin entrenamiento | — | — | — | Línea base |
| 1 | **DQN**: NatureCNN → Q(s,·) ∈ ℝ⁶, ε-greedy de 1.0 a 0.01 en el primer 10 %, buffer de 30k (por RAM), `learning_starts` 20k, red objetivo cada 1,000 pasos, pérdida Huber, Adam 1e-4, γ 0.99, batch 32, 1 entorno | 300k | 4 | 21:15 | Costo real de DQN en CPU (pasos/s, RAM) frente a PPO |
| 2 | **PPO** (8 entornos) sobre el v5 "tal cual": frameskip 4 interno **sin** max-pool y **sin** EpisodicLife | 500k | 4 | 21:30 | Efecto del preprocesamiento (se compara con Iter 3 a 500k) |
| 3 | **PPO final**: contrato completo (max-pool + EpisodicLife + clip), lr 2.5e-4 → 0 lineal, clip 0.1 → 0 lineal, entropía 0.01 | 2M | 8 | 22:05 | Modelo que se entrega |

**Hiperparámetros de PPO (rl-zoo, Atari):** `n_envs=8, n_steps=128, n_epochs=4, batch_size=256, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, learning_rate=lin(2.5e-4), clip_range=lin(0.1)`.

**Red:** `CnnPolicy` con NatureCNN compartida (Conv 32@8×8/4 → 64@4×4/2 → 64@3×3/1 → FC 512, ReLU) y dos cabezas, π con 6 salidas y V con 1. Tiene ≈ 1.69 M parámetros.

Si después de liberar memoria quedan menos de 4 GB libres, Iter 2 se corre **después** de Iter 1 y no en paralelo.

---

## 4. Cronograma de hoy

| Hora | Fase | Tareas | Listo cuando… |
|---|---|---|---|
| 19:30–20:00 | **F0 Preparación** | Cerrar apps pesadas (ChatGPT, ventanas extra de VS Code) y evitar que la laptop se suspenda; crear venv e instalar dependencias; copiar la base del Lab 5; escribir `entorno.py`, `configs/` y `entrenar.py`; smoke test de 20k pasos para medir FPS | El smoke test genera `progress.csv` y un checkpoint, y `pytest` pasa |
| 20:00–20:05 | **F1 Lanzamiento** | Lanzar Iter 3, Iter 1 e Iter 2 en segundo plano; evaluar Iter 0 (5 episodios) | Los 3 `progress.csv` crecen |
| 20:05–21:00 | **F2 EDA + informe 2.1/2.2** | Correr `analizar_entorno.py` (30 episodios aleatorios y figuras); redactar 2.1 y 2.2; cada 20 min anotar FPS, RAM y anomalías en la bitácora | Figuras del EDA y texto de 2.1–2.2 |
| 21:00–22:05 | **F3 Scripts de entrega** | Escribir `evaluar.py`, `generar_video.py`, `graficar.py`, `generar_informe.py` y el README; probarlos con un checkpoint intermedio de Iter 3; evaluar Iter 1 (21:15) e Iter 2 (21:30) | Un video de prueba y un JSON de evaluación salen sin pasos manuales |
| 22:05–22:25 | **F4 Resultados finales** | Evaluar `final` y `best_model` de Iter 3 con las mismas semillas y elegir uno → `modelos/`; evaluación oficial con video; `graficar.py`; ver el video y anotar comportamientos con minuto:segundo | MP4, JSON, figuras y tabla de iteraciones |
| 22:25–22:45 | **F5 Informe** | Completar 2.3–2.6 con los números reales; generar DOCX y PDF y revisarlo | PDF revisado |
| 22:45–22:55 | **F6 GitHub** | `pytest`; revisar `git status` (sin venv ni checkpoints); commit y push; verificar en github.com | README, pesos, video y PDF visibles en el repo |
| 22:55–23:00 | **Entrega** | Subir el PDF, el MP4 y el enlace del repo a la plataforma | Entregado |

**Regla de prioridad si algo se atrasa:** (1) que Iter 3 esté entrenando, (2) evaluación y video del mejor checkpoint, (3) informe, (4) repo. Si a las 20:15 no se ha lanzado nada, se lanza solo Iter 3 y se recortan Iter 1 e Iter 2 a lo que alcance.

> F0 asume que el código se escribe rápido (≈ 25 min). Si lo escribes a mano, lanza primero `entorno.py` + `entrenar.py` con lo mínimo y completa lo demás mientras entrena: cada 10 min de retraso son ~150–200k pasos menos.

---

## 5. Especificación de cada pieza

### `entorno.py`
- `CONFIG_FINAL` (dict serializable) con `env_id`, `make_kwargs`, `preprocesamiento` y `entrenamiento`.
- `envolver(env, cfg, entrenamiento, monitor_path=None)` aplica la pila de la sección 1.3.
- `crear_entorno_entrenamiento(cfg, seed, monitor_path)` y `crear_vec_entrenamiento(cfg, n_envs, seed, run_dir)` → `DummyVecEnv`.
- `crear_entorno_evaluacion(cfg, video_folder=None, name_prefix=...)` usa `crear_entorno` del Lab 5 (para poder grabar) + `envolver(..., entrenamiento=False)`.

### `agentes_ale.py` (Lab 5)
- Se copia tal cual. Único cambio: `generar_video_agente(..., envolver=None, fps=None)` para insertar el preprocesamiento sobre el entorno que graba. Con `frameskip=1` en el entorno base, grabar a 60 fps equivale a tiempo real.

### `agente.py`
- `cargar_modelo(ruta_zip)` → `(modelo, cfg)`: lee el JSON hermano y detecta si es PPO o DQN.
- `politica_greedy(modelo)` → `lambda obs, env: int(modelo.predict(obs, deterministic=True)[0])`, con la firma de función de agente del Lab 5.

### `scripts/entrenar.py`
- Argumentos: `--config`, `--threads` (`torch.set_num_threads`), `--timesteps`, `--hasta HH:MM`, `--seed 23236` y `--continuar ruta.zip` (reanudar).
- Logger de SB3 → `stdout`, `csv` y `tensorboard` en `resultados/<iter>/`.
- Callbacks:
  - `CheckpointCallback` cada 100k pasos.
  - `EvalCallback` cada 250k pasos: 5 episodios, greedy, entorno de evaluación; guarda `best_model.zip` y `evaluations.npz`.
  - Límite de reloj: detiene el entrenamiento y guarda.
  - Monitor de valores sobre 500 estados fijos (V(s) en PPO, max Q en DQN) para detectar divergencia.
- Al terminar guarda `final.zip` y `config.json` (contrato, hiperparámetros, pasos alcanzados, duración y versiones).

### `scripts/evaluar.py` (el que se corre en la presentación)
- `--modelo modelos/ppo_space_invaders_final.zip --episodios 5 --seed 23236 [--video]`.
- Usa `ejecutar_episodio` del Lab 5 con `max_steps=None` y la política greedy. Imprime una tabla por episodio con media, desviación y **máximo**, y guarda un JSON en `resultados/evaluaciones/`.

### `scripts/generar_video.py`
- Graba los 5 episodios del protocolo oficial con `generar_video_agente` y copia el de mayor puntaje a `entregables/space_invaders_ppo_final.mp4`.
- Escribe `metricas_video.json` igual que en el Lab 5: puntajes de los 5, semilla del elegido, frames, fps, duración, sha256 y versiones.

### `scripts/analizar_entorno.py` (EDA → sección 2.1)
- Espacios, significado de las acciones, vidas y `spec.kwargs`.
- 30 episodios aleatorios: puntaje, duración, % de pasos con recompensa 0, valores de recompensa, pasos entre recompensas y puntos por vida.
- Figuras:
  - frame RGB 210×160 frente a 84×84 en gris;
  - los 4 frames apilados;
  - frame sin max-pool frente a con max-pool (proyectiles);
  - histograma de recompensas.

### `scripts/graficar.py` (sección 2.3)
- Recompensa por episodio vs. pasos (cruda + media móvil de 100) de cada iteración, desde `monitor/*.csv`.
- Evaluación greedy vs. pasos, desde `evaluations.npz`.
- Pérdidas y diagnósticos:
  - PPO: `train/loss`, `value_loss`, `policy_gradient_loss`, `entropy_loss`, `approx_kl`, `clip_fraction`;
  - DQN: `loss` y `exploration_rate`;
  - el monitor de V/Q.
- Comparativas: Iter 1 vs. Iter 3 (mismo tiempo de reloj y mismos pasos) e Iter 2 vs. Iter 3 a 500k.

### `tests/test_entorno.py`
- La observación es `(4, 84, 84) uint8` en entrenamiento y en evaluación, y la acción es `Discrete(6)`.
- En evaluación, la recompensa es cruda (sin clip) y el episodio termina solo al perder las 3 vidas.
- En entrenamiento, la recompensa ∈ {−1, 0, 1} y hay `terminated` al perder una vida.
- El JSON del contrato hace ida y vuelta, y `politica_greedy` devuelve una acción válida.

---

## 6. Trabajo escrito: qué va en cada sección y de dónde sale

| Sección | Contenido | Fuente |
|---|---|---|
| 2.1 Problema y entorno | Objetivo del juego. Recompensa por fila (5–30) y nave de comando. `terminated` (3 vidas perdidas o invasores llegan abajo) frente a `truncated` (108,000 frames). Espacios de observación y acción. Recompensa dispersa (98 % ceros) → clip; sin reward shaping (justificar). Decisiones: 6 acciones (no `full_action_space`), max-pool por el parpadeo, sticky actions, vidas → EpisodicLife solo en entrenamiento. Baseline aleatorio. | `analizar_entorno.py`, figuras del EDA |
| 2.2 Metodología | DQN vs. Double/Dueling vs. PPO vs. A2C → PPO, con evidencia. Tabla de la NatureCNN con dimensiones: 4×84×84 → 32×20×20 → 64×9×9 → 64×7×7 → 3136 → 512 → 6 / 1. Exploración: entropía en PPO, ε-greedy en DQN. Tabla de hiperparámetros: pérdida, optimizador, lr, buffer, batch, red objetivo, γ, etc. | `configs/*.json` |
| 2.3 Iteraciones | Tabla de Iter 0–3: cambios, pasos, tiempo, recompensa de entrenamiento (media de los últimos 100 episodios) y de evaluación greedy (media ± desv., máx.). Curvas. Problemas encontrados y cómo se resolvieron, con su gráfica. | `resultados/`, `graficar.py`, bitácora |
| 2.4 Discusión | Qué cambio pesó más y por qué. Análisis cualitativo del video: estrategias y dónde falla, con tiempos. Limitaciones: solo CPU, ~2 M pasos frente a los 10–50 M de la literatura, buffer chico en DQN. Exploración vs. explotación: caída de la entropía y de ε, greedy vs. estocástico. | Video, curvas |
| 2.5 Conclusiones | Puntaje final e interpretación frente al aleatorio. Aprendizajes. Trabajo futuro: más pasos en GPU, Double/Dueling/PER/Rainbow, QR-DQN, escalar la recompensa en vez de recortarla, búsqueda de hiperparámetros. | — |
| 2.6 Repositorio | Enlace, cómo reproducir y cómo cargar los pesos | README |

**Referencias:**
- Mnih et al. 2015 (DQN, *Nature*).
- Schulman et al. 2017 (PPO).
- Machado et al. 2018 (*Revisiting the ALE*, sticky actions).
- van Hasselt et al. 2016 (Double DQN).
- Wang et al. 2016 (Dueling).
- Huang et al. 2022 (*The 37 Implementation Details of PPO*).
- Raffin et al. 2021 (SB3, *JMLR*).
- Documentación de Gymnasium y ALE.

---

## 7. Entregables finales (checklist)

- [ ] `entregables/Proyecto2_Informe_Ian_Cumes_23236.pdf` (+ `.docx` editable) con las secciones 2.1–2.6
- [ ] `entregables/space_invaders_ppo_final.mp4`: episodio completo con la misma configuración que la evaluación en vivo
- [ ] `entregables/metricas_video.json`: puntajes de los 5 episodios, sha256, frames y fps
- [ ] `modelos/ppo_space_invaders_final.zip` + `.json` (pesos + contrato del entorno)
- [ ] `scripts/evaluar.py` listo para correr sin pasos manuales (5 episodios, greedy, media y máximo)
- [ ] README con instalación, reproducción de cada iteración, carga de pesos y comandos de la presentación
- [ ] Repo público actualizado: https://github.com/iancumes/Proyecto2DeepLearning
- [ ] `pytest` en verde

**Para subir a la plataforma:** PDF + MP4 + enlace del repo.

---

## 8. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| Poca RAM (1.6 GB libres) | Cerrar apps; usar `DummyVecEnv`; buffer de DQN de 30k (~0.85 GB); si no alcanza, correr Iter 2 después de Iter 1 |
| La laptop se suspende o se desconecta | Cargador conectado, tapa abierta y suspensión desactivada mientras entrena |
| El entrenamiento se corta | Checkpoints cada 100k; `--continuar` reanuda desde el último |
| Preprocesamiento distinto en evaluación | Un solo `entorno.py`, JSON junto a los pesos y tests |
| Frameskip duplicado (16 frames) | `frameskip=1` en `gym.make`; `AtariPreprocessing` falla si no se cumple |
| Iter 3 no aprende (`ep_rew_mean` ≈ aleatorio a 1 M) | Revisar forma de la observación, frameskip y clip con el smoke test; usar `best_model` |
| El video no se escribe | `env.close()` garantizado (Lab 5); verificar el MP4 con OpenCV |
| Word no convierte a PDF | Abrir el DOCX y usar "Guardar como PDF" a mano |
| Archivos pesados en git | `.gitignore` para `checkpoints/`, `tb/` y `.venv/`; solo se sube el modelo final (~20 MB) |

---

## 9. Después de las 23:00 (competencia)

La competencia toma el **máximo de 5 episodios**, así que más pasos de entrenamiento significan mejor ranking.

1. Continuar Iter 3 desde `final.zip` durante la noche (`entrenar.py --continuar ... --timesteps 8000000`) o en Kaggle/Colab con GPU (3–4× más rápido).
2. Reevaluar con el mismo protocolo, regenerar el video con el mismo script y actualizar `modelos/` y el README.
3. Ensayar la demo: `python scripts\evaluar.py --episodios 5 --video` debe funcionar en un clon limpio del repo.

---

## 10. Comandos (PowerShell, desde la carpeta del proyecto)

```powershell
# F0 — entorno virtual y dependencias
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# F0 — smoke test (~1 min)
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter3_ppo_final.json --timesteps 20000 --run smoke

# F1 — lanzar iteraciones (una terminal por comando)
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter3_ppo_final.json --threads 8 --hasta 22:05
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter1_dqn.json --threads 4 --hasta 21:15
.\.venv\Scripts\python.exe scripts\entrenar.py --config configs\iter2_ppo_v5_directo.json --threads 4 --hasta 21:30
.\.venv\Scripts\tensorboard.exe --logdir resultados

# F4 — evaluación oficial y video
.\.venv\Scripts\python.exe scripts\evaluar.py --modelo modelos\ppo_space_invaders_final.zip --episodios 5 --seed 23236
.\.venv\Scripts\python.exe scripts\generar_video.py --modelo modelos\ppo_space_invaders_final.zip --seed 23236
.\.venv\Scripts\python.exe scripts\graficar.py

# F5 / F6 — informe, pruebas y GitHub
.\.venv\Scripts\python.exe scripts\generar_informe.py
.\.venv\Scripts\python.exe -m pytest
git add -A
git commit -m "Proyecto 2: agente PPO para Space Invaders, informe y video"
git push origin main
```
