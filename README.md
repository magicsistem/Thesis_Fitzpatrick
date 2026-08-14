# Thesis Fitzpatrick

Proyecto reproducible para segmentación de lesiones dermatoscópicas, derivación
de piel limpia y auditoría posterior por tipo Fitzpatrick. Compara los 15
backends neuronales registrados en `configs/segmentation_models.json` y S16
GrabCut, tanto en modo nativo (A) como con el frente común P0 (B1/B2 y C0–C3).
Fitzpatrick describe respuesta al sol; no se infiere de RGB, CIELAB o ITA.

## Estado y límites científicos

El código implementa P0 (FOV, vello, YOLOv3, ROI), S01–S16, métricas, folds,
entrenamiento B2, test sellado, anotación y revisión web. Los checkpoints
nativos públicos están identificados por SHA-256. Los cinco YOLO y los 75
checkpoints B2 todavía deben entrenarse y validarse en CEDIA; por ello no existe
aún ranking científico B2 ni se debe ejecutar el test sellado. Las 93 imágenes
Fitzpatrick siguen sin ground truth adjudicado y solo permiten revisión visual.

Verificado localmente: suite CPU, sintaxis, configuraciones, contratos y dry
runs. Validable sin GPU: manifests, folds, P0, S16 y web. Pendiente en CEDIA:
bootstrap dentro de la SIF, compilación Darknet CUDA, smoke AViT en A100 y jobs
SLURM. No se afirma una prueba A100 desde la laptop.

## Arquitectura

- `configs/`: catálogo científico, datasets y manifiesto externo CEDIA.
- `scripts/adapters/`: contratos ejecutables S01–S15; S16 vive en el paquete.
- `src/thesis_fitzpatrick/`: P0, FOV, GrabCut, métricas, datasets y artefactos.
- `scripts/benchmark/`: datasets, folds, YOLO, A/B1/B2, C0–C3 y sellado.
- `scripts/hpc/`: bootstrap, wrapper, preflight, smoke y plantillas SLURM.
- `web/` y `scripts/serve_segmentation_review.py`: revisión histórica y benchmark.
- `data/`, `models/`, `results/`: recursos locales persistentes ignorados por Git.
- `docs/phases/`: handoffs cerrados e inmutables por fase.

El contrato completo está en [docs/BENCHMARK_16_METHODS.md](docs/BENCHMARK_16_METHODS.md)
y la receta operacional en [docs/CEDIA_FROM_ZERO.md](docs/CEDIA_FROM_ZERO.md).
La recuperación tras una señal externa y la prueba obligatoria de un solo fold
están en [docs/CEDIA_HPC_RECOVERY.md](docs/CEDIA_HPC_RECOVERY.md).

## Requisitos confirmados de CEDIA

Acceso por <https://hpc.cedia.edu.ec> mediante Open OnDemand, Apptainer 1.3.0,
Slurm 22.05.5 y `$HOME/pytorch_24.01-py3.sif`. La SIF esperada mide
`10135441408` bytes y su SHA-256 es
`b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3`.
Contiene Python 3.10.12, PyTorch 2.2.0a0, torchvision 0.17.0a0 y CUDA 12.3.
El manifiesto auditable es `configs/hpc/external_resources.json`.

La SIF es de solo lectura. El bootstrap crea `.cedia/venv` con
`--system-site-packages`; `configs/hpc/core-constraints.txt` impide que pip
reemplace torch, torchvision o numpy. Debido a la ausencia de `squashfuse`, el
wrapper agrupa cada etapa en una sola ejecución de Apptainer y evita docenas de
conversiones de la SIF.

## Open OnDemand y clon limpio

Desde la terminal que muestre el portal (el nombre del menú puede variar):

```bash
cd "$HOME"
git clone https://github.com/magicsistem/Thesis_Fitzpatrick.git
cd "$HOME/Thesis_Fitzpatrick"
git switch chore/repository-audit-after-phase0
git rev-parse HEAD
bash scripts/hpc/inspect_cedia_environment.sh 2>&1 | tee "cedia_environment_$(date +%Y%m%d_%H%M%S).txt"
```

Para validar GPU desde una Interactive App, solicite partición `interactive`,
16 horas, 32 CPU, 96 GB RAM, una GPU y A100 SXM4 40 GB. Esos recursos son para
diagnóstico interactivo, no la reserva por defecto de cada entrenamiento.

## Bootstrap idempotente

Revise primero el plan y luego ejecute una sola etapa contenedorizada:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
bash scripts/hpc/bootstrap_cedia.sh --dry-run
bash scripts/hpc/bootstrap_cedia.sh --execute
python3 scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints
```

Se clonan commits fijados, se descargan checkpoints oficiales con reanudación,
se verifican hashes y se compila Darknet para compute capability 8.0. Repetir el
script reutiliza recursos correctos y falla ante un staging o hash sospechoso.
No modifica la SIF.

## Datasets y recursos manuales

ISIC 2018 e IMA++ tienen descargadores reanudables y requieren confirmación de
licencia. Este bloque obtiene ISIC, extrae sin extraer el ground truth sellado,
crea manifests y folds exclusivamente desde training:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1"
bash scripts/hpc/prepare_isic2018_cedia.sh --dry-run
bash scripts/hpc/prepare_isic2018_cedia.sh --execute
```

La etapa agrupada también obtiene metadatos oficiales, verifica archivos,
audita solapamientos y genera folds. IMA++ v1.1 es CC BY-NC-ND 4.0 y solo evaluación externa. Novice es robustez
auxiliar y queda deshabilitado. Si el portal o licencia exige descarga manual,
use Open OnDemand Files y coloque exactamente los recursos en los destinos del
manifiesto; después ejecute `manage_datasets.py import-*` y `verify`. La cohorte
Fitzpatrick requiere su manifest/metadata exacto y anotación humana, nunca datos
inventados. No se necesita SSH, rsync ni una copia de la laptop.

## Preflight, pruebas y smoke

Desde una sesión interactiva con A100:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1"
export TRAIN_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train.json"
bash scripts/hpc/preflight_cedia.sh --scope smoke --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT"
scripts/hpc/run_in_container.sh -- python -m unittest discover -s tests -v
scripts/hpc/run_in_container.sh -- python scripts/hpc/smoke_test.py \
  --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" --device cuda
```

El smoke carga el checkpoint real S01/AViT, separa carga de modelo e inferencia,
hace warm-up y tres repeticiones sincronizadas, escribe una máscara binaria a
resolución original y registra RAM/VRAM. Una salida vacía o saturada se conserva
con advertencia; no es un resultado científico.

## YOLO, B2 y benchmark mediante Slurm

Primero prepare las etiquetas/CFG de los cinco folds en una única ejecución del
contenedor; las listas deben regenerarse en CEDIA porque Darknet guarda rutas
absolutas. Luego envíe el array de una GPU por tarea:

```bash
cd "$HOME/Thesis_Fitzpatrick"
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export TRAIN_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train.json"
export FOLDS_FILE="$DATA_ROOT/manifests/isic2018_task1_train_folds_5.json"
export DARKNET_GPU_CONFIRMED=YES
bash scripts/hpc/prepare_yolo_folds_cedia.sh --dry-run
bash scripts/hpc/prepare_yolo_folds_cedia.sh --execute
YOLO_JOB=$(sbatch --parsable scripts/hpc/train_yolo_cedia.slurm)
echo "YOLO array: $YOLO_JOB"
```

Después del entrenamiento, infiera sobre cada fold de validación, seleccione
parámetros solo allí, congele los cinco YOLO y genere P0 out-of-fold:

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

Para A, B1 y C0–C3 completos sobre validation, configure un detector congelado
en una copia local de `configs/benchmark/default.json` y envíe:

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

Tras completar y revisar las 75 tareas B2, congele/evalúe B2 mediante una
dependencia `afterok` o envíe manualmente `scripts/hpc/finalize_b2_cedia.slurm`
con `DEVELOPMENT_MANIFEST` y `BENCHMARK_CONFIG` exportados. Nunca use `afterany`.

Cada backend usa una sola GPU; no hay DDP ni DataParallel. YOLO mapea tareas
0–4 a folds 0–4. B2 mapea `task // 5` a S01–S15 y `task % 5` al fold. B2 se
bloquea sin cinco YOLO congelados. El test sellado no participa.

## Monitoreo, desconexión y reanudación

```bash
squeue -u "$USER"
sacct -X -u "$USER" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -f slurm-thesis-yolo-JOB_TASK.out
find results/benchmark_v1 -type f -name '*.log' -o -name 'run_manifest.json' | sort
```

Cerrar el navegador no cancela un job Slurm. YOLO conserva backups y acepta
`RESUME_WEIGHTS`; B2 detecta `training_state.pt` y añade `--resume`. Los runs de
evaluación completados no se sobrescriben; uno incompleto debe auditarse y luego
continuarse con un nuevo `RUN_PREFIX`, conservando el anterior como evidencia.
No oculte OOM, timeout o adaptadores fallidos del denominador.

## Resultados, web y métricas

Los artefactos viven en `results/benchmark_v1/{preprocessing,runs,yolo,b2}`.
Registran Git, configuración, semillas, checkpoints, fallos, FOV, vello, ROI,
tiempos de P0/backend/end-to-end, CPU/RAM y VRAM. El tiempo de adaptador incluye
arranque/carga del subproceso y se etiqueta como tal; un sidecar separa carga y
forward sincronizado, con warm-up dentro del mismo proceso. El benchmark usa warm-up 1 y tres
repeticiones por defecto en CEDIA.

Desde una terminal de la Interactive App:

```bash
cd "$HOME/Thesis_Fitzpatrick"
scripts/hpc/run_in_container.sh --cpu -- python scripts/serve_segmentation_review.py \
  --host 0.0.0.0 --port 8000
```

Abra la URL proxy que ofrezca Open OnDemand, no exponga el puerto públicamente.
La web conserva el revisor histórico y muestra runs, artefactos, fallos,
fallbacks y métricas. P0 conserva máscara FOV cruda/corregida; toda máscara final
común se intersecta con FOV y la piel limpia excluye fuera de FOV, lesión y
vello. Los tests sintéticos cubren piel oscura sin marco, FOV circular y bordes.

## Solución de problemas y limpieza segura

- `Converting SIF file to temporary sandbox`: esperado sin `squashfuse`; no
  fragmente una etapa en muchos `apptainer exec`.
- hash SIF distinto: no use `--allow-unverified-sif` hasta documentar y probar
  la nueva imagen.
- importación fallida: no reinstale torch/numpy; revise `.cedia/venv` y
  `python -m pip check` dentro del wrapper.
- checkpoint inválido: conserve el archivo para auditoría fuera de la ruta
  canónica y reanude la descarga; nunca omita SHA-256.
- OOM/timeout: conserve logs/estado y reduzca batch o reenvíe con más tiempo;
  no cambie parámetros científicos silenciosamente.

Puede retirar manualmente solo `.partial` verificados como descargas
interrumpidas y sandboxes temporales creados por el runtime después de que no
haya jobs activos. Nunca borre `data/`, `models/`, `results/`, checkpoints,
manifests congelados ni evidencia de runs con comandos amplios.

## Reproducibilidad, licencias y política Git

El manifiesto fija commits, URLs, licencias, tamaños y hashes. AViT y
BA-Transformer no declaran licencia upstream; se descargan desde el autor para
uso de investigación y no se redistribuyen. IMA++ conserva CC BY-NC-ND 4.0;
ISIC conserva términos y atribución por imagen. Darknet mantiene atribución
upstream. Consulte `docs/SEGMENTATION_CHECKPOINT_AUDIT.md`.

Git solo contiene código, pruebas, configuración y documentación. `.cedia/`,
datos, modelos, pesos, SIF, resultados, logs Slurm y diagnósticos CEDIA quedan
ignorados. No guarde secretos ni configuración privada del portal.
