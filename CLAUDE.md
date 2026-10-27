# fedecg-lab

Clasificación multietiqueta de ECG de 12 derivaciones (PTB-XL) para medir qué cuesta entrenar con datos que no salen del hospital: aprendizaje federado, privacidad diferencial y explicabilidad frente a una línea base centralizada.

## Seguimiento

- Estado, prioridad y roadmap viven en el dashboard git (~/git/.project-dashboard, http://localhost:1403); no los dupliques aquí.
- Al terminar una sesión de trabajo, actualiza NEXT.md con el siguiente paso: el dashboard lo lee.

## Stack

- Python 3.11+ gestionado con uv (`uv.lock` exacto): PyTorch, Flower (FedAvg/FedProx), Opacus (DP-SGD), MLflow, WFDB.
- Paquete en `src/fedecg/` (data, models, training, federated, privacy, explain); experimentos como YAML en `configs/` con herencia `extends`.
- Dashboard en `app/` (Vite + React + TypeScript, Node 22+); notebooks 01-06 en `notebooks/`.
- Calidad: ruff (lint y format), pytest, pre-commit (incluye nbstripout). CI en `.github/workflows/ci.yml`.

## Comandos

- Instalar: `uv sync` (dashboard: `npm --prefix app install`).
- Datos PTB-XL: `uv run python scripts/download_data.py`.
- Entrenar: `uv run python scripts/train_centralized.py [--config smoke.yaml]`, `uv run python scripts/train_federated.py --config <cfg>.yaml`, `uv run python scripts/tune.py --config <cfg>.yaml`; todo el pipeline: `scripts/reproduce.sh [fases]` (~3,5 h).
- Arrancar dashboard: `uv run python scripts/export_dashboard.py`, luego `npm --prefix app run dev` (http://localhost:5173); servidor de entrenamiento local: `uv run python scripts/serve.py`.
- Tests: `uv run pytest` (CI usa `-m "not slow and not needs_data"`); dashboard: `npm --prefix app run build`.
- Lint: `uv run ruff check .` y `uv run ruff format --check .`.

## Convenciones

- No commitear datos, checkpoints, runs ni secretos: `data/`, `checkpoints/`, `runs/`, `mlruns/`, `*.pt` y `app/public/data/` están ignorados (contienen waveforms de PTB-XL).
- Solo se versionan las tablas y figuras pequeñas de `results/`; cada ejecución añade una fila a `results/tables/experiments.csv`.
- Notebooks sin salidas (nbstripout); el código que funciona pasa de notebook a `src/fedecg/` con su test.
- Particiones con generadores NumPy explícitos, nunca estado aleatorio global. El CI nunca descarga datos ni entrena.

## Contexto

- Release 1.0.0 terminado (8 fases). Línea base: ResNet1D con GroupNorm (Opacus no soporta BatchNorm), AUROC macro 0,917 en el fold 10 (0,922 con 3 semillas; publicado 0,930), afinada solo sobre validación.
- Resultados: descentralizar cuesta 0,007-0,026 AUROC (5/10/20 hospitales); la heterogeneidad realista casi nada, solo el sesgo sintético extremo (~0,017) y FedProx no lo recupera; DP-SGD cuesta 0,024/0,035/0,066 (ε=8/3/1) y más por hospital; el modelo reconoce el infarto por el QRS, no por el ST.
- Los experimentos DP usan la red de 32 canales y su propia receta; el coste de privacidad se mide contra su gemelo sin ruido.
- Limitaciones: federación simulada en un solo proceso, datos de un único país, privacidad a nivel de registro, una semilla por ajuste (ruido ~0,003), saliency correlacional, sin validación clínica. No es un producto sanitario.
