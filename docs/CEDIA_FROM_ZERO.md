# CEDIA desde cero con Open OnDemand

Guía operacional para Bash en CEDIA. No requiere SSH, Conda remoto ni archivos
de la laptop. Los comandos largos se agrupan por etapa para reducir las
conversiones temporales de la SIF de 10 GB.

## Etapa 0 — Requisitos

**Propósito.** Confirmar portal, espacio, Slurm, GPU y la SIF sin modificarla.

Desde la aplicación de terminal que muestre Open OnDemand, o desde una
Interactive App con partición `interactive`, 16 horas, 32 CPU, 96 GB RAM y una
A100 SXM4 40 GB:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
bash scripts/hpc/inspect_cedia_environment.sh 2>&1 | \
  tee "cedia_environment_$(date +%Y%m%d_%H%M%S).txt"
sha256sum "$SIF_PATH"
stat -c '%n %s bytes' "$SIF_PATH"
```

**Resultado esperado.** Apptainer 1.3.0; SIF de 10135441408 bytes y SHA-256
`b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3`.
En una sesión GPU, PyTorch debe ver una A100. El diagnóstico queda ignorado por
Git.

**Condición para continuar.** Hash correcto, al menos 20 GiB libres y runtime
disponible. Si falla, no descargue nada: confirme la ruta observada con Files o
ejecute el diagnóstico con `SIF_PATH=/ruta/real`.

## Etapa 1 — Clonar o actualizar

**Propósito.** Obtener únicamente código versionado desde GitHub.

```bash
cd "$HOME"
if [[ -d Thesis_Fitzpatrick/.git ]]; then
  cd Thesis_Fitzpatrick
  git status --short --branch
  git pull --ff-only
else
  git clone https://github.com/magicsistem/Thesis_Fitzpatrick.git
  cd Thesis_Fitzpatrick
  git switch chore/repository-audit-after-phase0
fi
git rev-parse --show-toplevel
git log -1 --oneline
```

**Resultado esperado.** Raíz `$HOME/Thesis_Fitzpatrick`, rama indicada y el
commit publicado más reciente. **Condición para continuar:** árbol sin cambios
inesperados. Si `git pull --ff-only` se niega, conserve los archivos y revise
`git status`; no haga reset.

## Etapa 2 — Bootstrap

**Propósito.** Crear un venv persistente desde Python de la SIF, clonar commits
fijados, descargar/verificar pesos y compilar Darknet para A100. La SIF no se
modifica.

```bash
cd "$HOME/Thesis_Fitzpatrick"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
bash scripts/hpc/bootstrap_cedia.sh --dry-run
bash scripts/hpc/bootstrap_cedia.sh --execute 2>&1 | tee bootstrap_cedia.log
python3 scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints
```

**Resultado esperado.** `.cedia/venv`, siete fuentes en commits exactos, 16
archivos de pesos verificados y binario `models/yolov3-darknet/source/darknet`.
**Condición para continuar:** `bootstrap_resources.py` termina en cero. Si una
descarga se corta, repita la etapa: usa `.partial`/`gdown --continue`. Si un hash
es inválido, no lo acepte ni lo renombre como checkpoint válido.

## Etapa 3 — Datos automáticos y manuales

**Propósito.** Descargar ISIC 2018 oficialmente, no extraer la máscara sellada
de test, crear manifests con SHA-256 y folds sin fugas.

```bash
cd "$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1"
bash scripts/hpc/prepare_isic2018_cedia.sh --dry-run
bash scripts/hpc/prepare_isic2018_cedia.sh --execute
```

**Resultado esperado.** 2594/100/1000 imágenes, máscaras para train/validation,
test sin máscaras accesibles y cinco folds. **Condición para continuar:** todos
los imports tienen `complete=true`, `verify` y auditoría de folds pasan.

IMA++ se prepara por separado tras aceptar CC BY-NC-ND 4.0:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export IMAPP_ROOT="$HOME/Thesis_Fitzpatrick/data/external/imapp_v1_1"
bash scripts/hpc/prepare_imapp_cedia.sh --dry-run
bash scripts/hpc/prepare_imapp_cedia.sh --execute
```

Si Novice o la cohorte Fitzpatrick exigen interacción, use Open OnDemand Files
para subir los archivos oficiales a los destinos de
`configs/hpc/external_resources.json`; no sustituya datos. Registre Novice con
`register-novice` y Fitzpatrick con `register-fitzpatrick`. La anotación de las
93 imágenes sigue pendiente y no bloquea ISIC.

## Etapa 4 — Preflight y pruebas

**Propósito.** Validar en una sola conversión de SIF paquetes, GPU, fuentes,
pesos, datos, permisos y configuración.

```bash
cd "$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1"
export TRAIN_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train.json"
bash scripts/hpc/preflight_cedia.sh --scope smoke --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT"
scripts/hpc/run_in_container.sh -- python -m unittest discover -s tests -v
```

**Resultado esperado.** JSON `passed=true` y suite sin fallos. **Condición para
continuar:** cero errores; las advertencias explican recursos opcionales. Si
falla un import, no actualice torch/numpy: revise el venv y repita bootstrap.

## Etapa 5 — Smoke real

**Propósito.** Probar AViT/checkpoint/GPU/máscara sin ejecutar el benchmark.

Interactivo:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1"
scripts/hpc/run_in_container.sh -- python scripts/hpc/smoke_test.py \
  --manifest "$DATA_ROOT/manifests/isic2018_task1_train.json" --data-root "$DATA_ROOT" --device cuda
```

O mediante Slurm:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export SMOKE_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train.json"
sbatch scripts/hpc/smoke_cedia.slurm
```

**Resultado esperado.** `results/hpc_smoke/.../lesion_mask.png` y
`smoke_report.json` con carga, inferencia, RAM y VRAM. **Condición para
continuar:** contrato binario y dimensiones pasan; una máscara vacía/saturada
es advertencia científica que debe revisarse, no ocultarse.

## Etapa 6 — YOLO, B2 y benchmark

**Propósito.** Ejecutar trabajos largos reanudables, una GPU por tarea, sin test.

Prepare los cinco folds en una sola ejecución del contenedor antes de enviar
YOLO:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1"
export TRAIN_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train.json"
export FOLDS_FILE="$DATA_ROOT/manifests/isic2018_task1_train_folds_5.json"
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick" SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export DARKNET_GPU_CONFIRMED=YES
bash scripts/hpc/prepare_yolo_folds_cedia.sh --dry-run
bash scripts/hpc/prepare_yolo_folds_cedia.sh --execute
YOLO_JOB=$(sbatch --parsable scripts/hpc/train_yolo_cedia.slurm)
echo "YOLO array: $YOLO_JOB"
```

Tras terminar los cinco entrenamientos, finalice los folds, prepare P0
out-of-fold (cada imagen usa el YOLO que no la vio) y solo entonces envíe B2:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export TRAIN_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train.json"
export FOLDS_FILE="$DATA_ROOT/manifests/isic2018_task1_train_folds_5.json"
export YOLO_FROZEN_ROOT="$PROJECT_ROOT/results/benchmark_v1/yolo"
FINALIZE_YOLO_JOB=$(sbatch --parsable scripts/hpc/finalize_yolo_cedia.slurm)
P0_JOB=$(sbatch --parsable --dependency="afterok:$FINALIZE_YOLO_JOB" scripts/hpc/prepare_p0_oof_cedia.slurm)
B2_JOB=$(sbatch --parsable --dependency="afterok:$P0_JOB" scripts/hpc/train_b2_cedia.slurm)
echo "YOLO finalize=$FINALIZE_YOLO_JOB P0=$P0_JOB B2=$B2_JOB"
```

Ejecute A/B1/C0–C3 completos solo con una configuración CEDIA que apunte al
detector congelado de desarrollo:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export DEVELOPMENT_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_validation.json"
scripts/hpc/run_in_container.sh -- python scripts/hpc/configure_benchmark_cedia.py \
  --frozen-yolo-root "$PROJECT_ROOT/results/benchmark_v1/yolo"
export BENCHMARK_CONFIG="$PROJECT_ROOT/.cedia/benchmark.cedia.json"
sbatch scripts/hpc/run_benchmark_cedia.slurm
```

Después de revisar que las 75 tareas B2 terminaron correctamente:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export DEVELOPMENT_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_validation.json"
export BENCHMARK_CONFIG="$PROJECT_ROOT/.cedia/benchmark.cedia.json"
B2_FINAL_JOB=$(sbatch --parsable scripts/hpc/finalize_b2_cedia.slurm)
echo "B2 freeze and evaluation: $B2_FINAL_JOB"
```

**Resultado esperado.** Cinco YOLO congelados antes de B2, 75 checkpoints B2
antes de `freeze_b2.py`, y seis runs completos de desarrollo. **Condición para
continuar:** manifests/hashes válidos y cero fuga. Si se interrumpe, reenvíe:
YOLO toma el backup más nuevo; B2 toma `training_state.pt`. Un run de evaluación
incompleto se conserva y exige un `RUN_PREFIX` nuevo después de auditarlo.

## Etapa 7 — Revisar y recuperar resultados

**Propósito.** Monitorear sin depender de una pestaña abierta y revisar la web.

```bash
squeue -u "$USER"
sacct -X -u "$USER" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
find "$HOME/Thesis_Fitzpatrick/results/benchmark_v1" -type f \
  \( -name 'run_manifest.json' -o -name 'report.json' -o -name '*.log' \) | sort
cd "$HOME/Thesis_Fitzpatrick"
scripts/hpc/run_in_container.sh --cpu -- python scripts/serve_segmentation_review.py \
  --host 0.0.0.0 --port 8000
```

**Resultado esperado.** Runs, artefactos P0, métricas, fallos y fallbacks en la
web mediante el proxy de Open OnDemand. **Condición para continuar:** resultados
copiados o descargados con Open OnDemand Files y checksums conservados. No
añada `results/` a Git.

## Etapa 8 — Continuar tras interrupción

**Propósito.** Recuperar estado persistente sin borrar evidencia.

```bash
cd "$HOME/Thesis_Fitzpatrick"
git status --short --branch
squeue -u "$USER"
sacct -X -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
find results/benchmark_v1/yolo -name '*.weights' -o -name 'frozen.json' | sort
find results/benchmark_v1/b2 -name 'training_state.pt' -o -name '*.metadata.json' | sort
python3 scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints
```

**Resultado esperado.** Fuentes/pesos intactos y estado reanudable visible.
Cerrar el portal no cancela Slurm. Si faltan archivos, repita bootstrap o la
descarga reanudable correspondiente; no use `git clean`, borrados recursivos ni
el test sellado para depurar.
