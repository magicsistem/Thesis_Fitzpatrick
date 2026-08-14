# Recuperación HPC de CEDIA

El launcher sólo envía el DAG. `preflight_pipeline_cedia.slurm` contiene el
bootstrap, preparación de ISIC, pruebas y preparación de folds; por ello el
launcher no necesita GPU ni debe durar más que los `sbatch` y la escritura
atómica de su registro.

## Diagnóstico antes de relanzar

No se borra `results/`, `.cedia/diagnostics/`, `backup/`, estados ni logs.
En CEDIA, desde el clon reparado, guarde la evidencia del job que falló:

```bash
sacct -j 23016 --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList
scontrol show job 23016
find results/benchmark_v1/yolo/fold-1 -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
```

Cada intento nuevo deja `training_state.json`, `training/attempts/` y
`.cedia/diagnostics/yolo-*-fold-*-attempt-*.log`. El monitor registra hora,
árbol PID/PGID, cgroup `memory.*`/`pids.*`, RAM, GPU/temperatura/potencia,
espacio/inodos/cuota y directorios temporales de Apptainer. Un log que termina
bruscamente sigue siendo evidencia de una terminación externa; no demuestra
OOM ni identifica por sí mismo al emisor de SIGKILL.

## Despliegue con Git antiguo

Revise y cree el commit localmente; no use `git switch`. Transfiera un patch
binario y aplíquelo con `git am` en CEDIA:

```bash
# equipo local, después de revisar y crear un único commit manual
git format-patch -1 --stdout > hpc-recovery.patch

# CEDIA
cd "$HOME/Thesis_Fitzpatrick"
git checkout fix/hpc-pipeline-validation
git am --3way hpc-recovery.patch
git rev-parse HEAD
git status --short
```

Si el commit ya está publicado, sustituya las tres últimas líneas por
`git fetch origin` y `git checkout <commit-verificado>`; confirme siempre que
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

1. Si el preflight pesado ya completó para el mismo commit/recursos, no lo
   repita. Ejecute la puerta ligera sin GPU ni entrenamiento:

   ```bash
   bash scripts/hpc/validate_existing_cedia.sh 1
   ```

   Crea y verifica los temporales exactos del job local, revisa recursos ya
   presentes y emite el fingerprint del checkpoint compatible. Sus dos
   comprobaciones Python se ejecutan mediante `run_in_container.sh` dentro de
   la SIF (nunca con el `python3` del nodo login); no llama bootstrap ni
   preparación de datos. Deténgase si no muestra `resume_weights_path`;
   entonces no se debe iniciar Darknet.
   Use `sbatch scripts/hpc/preflight_pipeline_cedia.slurm` sólo si esa puerta
   revela que faltan artefactos o si cambió el contrato.
2. Tras la puerta ligera, reanude **un solo fold que tenga checkpoint**, por
   ejemplo fold 1: `YOLO_MAX_ATTEMPTS=2 sbatch --array=1-1 scripts/hpc/train_yolo_cedia.slurm`.
   El selector no usa `mtime`: exige ruta `fold-1/backup`, nombre Darknet,
   cabecera/iteración, tamaño/hash, hashes de cfg/datos/pesos iniciales y el
   contrato anterior. Un cambio sólo de commit de la reparación es trazado y
   permitido; cualquier otro cambio se rechaza. Deténgase ante estado
   `failed`, `interrupted` sin inventario válido, un segundo `75`, o cualquier
   nueva señal externa: entregue los diagnósticos a CEDIA antes de escalar.
3. Si llega a 6000 y `training_state.json` es `completed`, ejecute
   `sbatch --dependency=afterok:<YOLO_FOLD_JOB> --array=1-1 scripts/hpc/finalize_yolo_cedia.slurm`.
   Deténgase si no aparecen `frozen.json`, su hash y la validación de fold.
4. Sólo después de validar el fold de prueba, envíe los restantes de forma
   controlada: `sbatch --array=0-4%2 scripts/hpc/train_yolo_cedia.slurm`, y
   luego `finalize_yolo_cedia.slurm`. La puerta de cinco folds impide P0 si
   falta uno, está incompleto o su hash cambia.
5. P0, B2 y benchmark se envían únicamente mediante las dependencias
   `afterok` del launcher. Deténgase tras P0 si `oof_manifest.json` o
   `oof_phase.json` no son `completed`; tras B2 si falta cualquiera de los
   75 metadata/checkpoints; y antes de benchmark si la configuración o los
   cinco `frozen.json` no validan.

No use `launch_all_cedia.sh` durante los pasos 1--3: envía el DAG completo por
diseño. Úselo sólo cuando los cinco folds estén congelados o cuando se quiera
crear una ejecución limpia completa. Su registro `.cedia/cedia_job_chain_*.txt`
se publica de forma atómica y no contiene una espera de smoke.
