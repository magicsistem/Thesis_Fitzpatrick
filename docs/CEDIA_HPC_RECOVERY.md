# Recuperación HPC de CEDIA

El launcher sólo envía el DAG. `preflight_pipeline_cedia.slurm` contiene el
bootstrap, preparación de ISIC, pruebas y preparación de folds; por ello el
launcher no necesita GPU ni debe durar más que los `sbatch` y la escritura
atómica de su registro.

## Recuperación granular de A/B1

La evaluación A/B1 tiene un reparador específico:
`scripts/benchmark/resume_evaluation.py`. No repite automáticamente las
combinaciones válidas. Primero audita los `result.json` y artefactos, separa
predicciones degeneradas de fallos técnicos y conserva el estado previo bajo
`resume_attempts/<attempt_id>/`.

La reanudación se rechaza si cambian dataset, split, orden de imágenes,
configuración, manifiesto o checkpoints; también si el run original estaba
dirty, el árbol actual está dirty o el commit original no es ancestro del
commit de reparación.

Para el A `cedia-development-a-23188`:

```bash
cd "$HOME/Thesis_Fitzpatrick"
git status --short
git rev-parse HEAD

export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export DEVELOPMENT_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_validation_disjoint.json"
export BENCHMARK_CONFIG="$PROJECT_ROOT/.cedia/benchmark.cedia.json"
export TARGET_RUN_ID=cedia-development-a-23188

sbatch scripts/hpc/resume_evaluation_cedia.slurm
```

El reparador detectará los fallos técnicos existentes y generará micro-runs
únicamente para esas combinaciones; los micro-runs terminados se archivan
dentro del propio run reparado. Si todos quedan válidos, reconstruye
`metrics_per_image.csv`, `failures.csv`, `quality_flags.csv`,
`metrics_summary.csv`, `metrics_summary.md`, `statistical_comparisons.csv` y
`report.json`, y cambia el manifest a `completed`.

Para B1 use:

```bash
export TARGET_RUN_ID=cedia-development-b1-23188
sbatch scripts/hpc/resume_evaluation_cedia.slurm
```

Si B1 no contiene fallos técnicos, no se repite ninguna inferencia: sólo se
reclasifican las predicciones degeneradas y se generan los reportes faltantes.

## Diagnóstico antes de relanzar

No se borra `results/`, `.cedia/diagnostics/`, `backup/`, estados ni logs.
En CEDIA, desde el clon reparado, guarde la evidencia del job que falló:

```bash
sacct -j 23016 --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList
scontrol show job 23016
find results/benchmark_v1/yolo/fold-1 -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
```

Cada tarea deja `training_state.json`, `training/attempts/` y
`.cedia/diagnostics/yolo-*-fold-*.log`. Una única invocación `singularity exec
--nv` (o `apptainer` sólo como fallback registrado) contiene la validación, la selección del checkpoint y hasta dos intentos
de Darknet; por tanto no se reconvierte la SIF entre reintentos. El monitor lee
atómicamente el PID/PGID real publicado por Python, además de registrar hora,
cgroup `memory.*`/`pids.*`, RAM, GPU/temperatura/potencia, espacio/inodos/cuota
y directorios temporales de Apptainer. Un log que termina
bruscamente sigue siendo evidencia de una terminación externa; no demuestra
OOM ni identifica por sí mismo al emisor de SIGKILL.

## Despliegue con Git antiguo

Revise y cree el commit localmente; no use `git switch`. Transfiera un patch
binario y aplíquelo con `git am` en CEDIA:

```bash
# equipo local, después de revisar y crear un único commit manual
git format-patch -1 --stdout > hpc-recovery.patch

# CEDIA: Git antiguo, actualice exactamente la referencia remota requerida.
cd "$HOME/Thesis_Fitzpatrick"
git checkout fix/hpc-pipeline-validation
git fetch origin \
  refs/heads/fix/hpc-pipeline-validation:refs/remotes/origin/fix/hpc-pipeline-validation
git merge --ff-only \
  refs/remotes/origin/fix/hpc-pipeline-validation
git rev-parse HEAD
git status --short
```

Si se usa el patch local en vez del remoto, sustituya únicamente `fetch` y
`merge` por `git am --3way hpc-recovery.patch`. Confirme siempre que
`git status --short` no imprima nada antes de enviar jobs.

## Validación escalonada obligatoria

Exporte una vez las rutas persistentes:

```bash
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export TRAIN_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json"
export DEVELOPMENT_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_validation_disjoint.json"
export FOLDS_FILE="$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json"
export YOLO_FROZEN_ROOT="$PROJECT_ROOT/results/benchmark_v1/yolo"
export DARKNET_GPU_CONFIRMED=YES
```

## Diagnóstico del runtime en un nodo Slurm

Antes de cualquier entrenamiento, envíe sólo esta comprobación CPU. Conserva
`$HOME/pytorch_24.01-py3.sif`, no instala paquetes y no toca checkpoints:

```bash
test -d "$PROJECT_ROOT" && test -f "$SIF_PATH" || { echo "PROJECT_ROOT/SIF_PATH inválidos" >&2; exit 2; }
sbatch "$PROJECT_ROOT/scripts/hpc/diagnose_container_runtime_cedia.slurm"
```

Interprete el stdout así: `kind=singularity` confirma el runtime preferido y
debe mostrar `sys_executable`, Python, torch, CUDA y `cuda_available` dentro
del SIF. `kind=apptainer-fallback` no cambia la SIF, pero mantiene el riesgo de
conversión temporal; no relance el array completo hasta revisar su coste. El
mensaje `Neither singularity nor apptainer` detiene el proceso antes de tocar
un checkpoint; consulte `module avail` registrado por el diagnóstico y use
sólo el módulo oficial que muestre CEDIA.

1. Si el preflight pesado ya completó para los mismos recursos, no lo repita.
   Determine los folds pendientes con `sacct`, `training_state.json` y los
   checkpoints existentes. No ejecute antes `validate_existing_cedia.slurm`:
   el launcher envía exactamente una validación CPU contenida y esa etapa
   compara la lista explícita con el estado real. El script independiente se
   conserva sólo para diagnóstico manual, no como requisito del relanzamiento.
2. `run_in_container.sh` crea un temporal Apptainer único por invocación y lo
   limpia al terminar el payload. La caché sigue siendo estable por usuario;
   no hay locks propios del pipeline. Los diagnósticos de entrenamiento los
   inicia el wrapper después de calcular esas rutas y registran PID/PGID del
   payload y del Darknet real publicado atómicamente. El monitor usa una sesión
   propia, verifica PID=PGID=SID antes de pararlo y sólo señaliza su PID exacto
   (y su timer hijo verificado); nunca envía una señal de grupo a Darknet.
   `PYTHONNOUSERSITE=1` y `THESIS_IN_CONTAINER=1` se exportan al payload:
   no se usa `~/.local`, y una llamada accidental al wrapper desde el SIF hace
   `exec` directo en vez de abrir un segundo contenedor. La identidad normal
   de la SIF es ruta, tamaño, mtime y propietario; SHA-256 completo queda para
   auditoría manual, no para cada fold.
3. Para recuperar la cadena rota por `afterok:23063`, confirme primero que el
   array es terminal y que los seis descendientes nunca iniciaron:

   ```bash
   sacct -j 23063 --format=JobID,State,ExitCode,Elapsed,NodeList
   squeue -j 23064,23065,23066,23067,23068,23069
   # Sólo si los seis están PENDING por Dependency y ninguno inició:
   scancel 23064 23065 23066 23067 23068 23069
   ```

   `scancel` es una decisión manual; el launcher nunca lo ejecuta. Después,
   el launcher no espera, no solicita GPU y **no envía preflight**. Rechaza la
   recuperación mientras el array previo o esos descendientes sigan activos.
   El validador es la primera dependencia, compara exactamente la lista y el
   freeze posterior procesa los cinco folds. Si fold 4 también quedó
   incompleto, use `0,3,4`; si validó completo, use `0,3`:

   ```bash
   YOLO_EXCLUDE_NODES=compute-0-1 \
   bash scripts/hpc/launch_pipeline_cedia.sh --recover-existing --skip-preflight \
     --folds 0,3,4 --previous-yolo-job 23063 \
     --blocked-jobs 23064,23065,23066,23067,23068,23069
   ```

   El manifiesto atómico queda en `.cedia/resume_pipeline_*.json`. Un segundo
   lanzamiento se rechaza si aún hay jobs de ese manifiesto o resultados
   descendientes existentes. Los IDs previos son parámetros explícitos, no
   valores permanentes hardcodeados por el launcher.
4. El selector no usa `mtime`: primero filtra por ruta `fold-N/backup`, nombre,
   archivo regular, tamaño y cabecera/iteración; sólo calcula SHA-256 completo
   del candidato ganador. Registra `hash_files_count`, `hash_bytes_total`,
   `validation_seconds`, `container_startup_seconds`, `training_seconds` y
   `retry_seconds`. También exige hashes de cfg/datos/pesos iniciales y el
   contrato anterior. Un cambio sólo de commit de la reparación es trazado y
   permitido; cualquier otro cambio se rechaza. Deténgase ante estado
   `failed`, `interrupted` sin inventario válido, un segundo `75`, o cualquier
   nueva señal externa: entregue los diagnósticos a CEDIA antes de escalar.
5. P0, B2 y benchmark se envían únicamente mediante las dependencias
   `afterok` del launcher. Deténgase tras P0 si `oof_manifest.json` o
   `oof_phase.json` no son `completed`; tras B2 si falta cualquiera de los
   75 metadata/checkpoints; y antes de benchmark si la configuración o los
   cinco `frozen.json` no validan.

## P0 OOF paralelo

P0 usa 24 procesos `spawn` cuando Slurm asigna 32 CPU. Cada proceso limita
OpenCV/BLAS a un hilo y crea una sola instancia CPU de su detector YOLO
congelado para el fold que procesa; no comparte `cv2.dnn.Net` ni usa CUDA de
OpenCV. Antes de crear pools comprueba que los cinco folds asignan una vez y
sólo una vez cada imagen de `train`. `--reuse-cache` conserva los directorios
completos existentes: el manifiesto de cada imagen se publica después de sus
artefactos mediante escrituras atómicas, y un directorio parcial se recalcula.
El `oof_manifest.json` normal sólo se publica al finalizar el conjunto completo
sin fallos.

## Límite de entrenamiento y diagnóstico de terminación

El límite metodológico efectivo es `max_batches=6000` (con pasos 4800 y 5400).
No hay en esta cadena un *early stopping* por mAP, pérdida, `patience` o una
meseta alrededor de 3000: esa cifra no es un límite de Darknet. `-map` se
mantiene exactamente como estaba en el comando de Darknet y no lo interpreta el
wrapper como criterio de parada. El único `75` que puede devolver YOLO es la
traducción explícita de una interrupción del hijo Darknet por señal; no es una
señal de convergencia ni un supuesto OOM.

No se añadió un "killer" de progreso. El proceso Python actualiza, cada 30 s,
un *heartbeat* y la última iteración observada leyendo sólo los últimos 256 KiB
del log; no mata el entrenamiento. Al terminar, el estado conserva PID/PGID
real, hora de inicio/parada, `returncode`, causa, checkpoint elegido, hashes y
contadores de bytes/archivos hasheados. El monitor sólo observa el PID publicado
del binario y su GPU; se detiene con el payload. Por tanto una próxima señal
externa deja una causa comprobable sin convertir una evaluación `-map` lenta en
un falso bloqueo.

Para seguimiento sin sesión persistente, sustituya `JOB` y `FOLD` por valores
reales y ejecute únicamente consultas:

```bash
PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
JOB=23063
FOLD=0
squeue -j "$JOB" -o '%.18i %.12T %.20R %.20M %.12N'
sacct -j "$JOB" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
STATE="$PROJECT_ROOT/results/benchmark_v1/yolo/fold-$FOLD/training/training_state.json"
LOG="$PROJECT_ROOT/.cedia/diagnostics/yolo-${JOB}-fold-${FOLD}.log"
test -r "$STATE" && sed -n '1p' "$STATE"
test -r "$LOG" && tail -n 80 "$LOG"
find "$PROJECT_ROOT/results/benchmark_v1/yolo/fold-$FOLD/backup" -maxdepth 1 -type f \
  -printf '%f %s bytes\\n' | sort
```

No use `launch_all_cedia.sh` durante los pasos 1--3: envía el DAG completo por
diseño. Úselo sólo cuando los cinco folds estén congelados o cuando se quiera
crear una ejecución limpia completa. Su registro `.cedia/cedia_job_chain_*.txt`
se publica de forma atómica y no contiene una espera de smoke.

Para revisar los jobs antiguos sin mantener una sesión:

```bash
squeue -j 23064,23065,23066,23067,23068,23069
sacct -j 23062,23063,23064,23065,23066,23067,23068,23069 --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
# Sólo después de revisar y decidirlo manualmente:
# scancel 23064 23065 23066 23067 23068 23069
```
