# Evidencia CEDIA y decisiones de ejecución

Este documento conserva la evidencia técnica confirmada el 2026-08-10. La guía
copiable de principio a fin está en [CEDIA_FROM_ZERO.md](CEDIA_FROM_ZERO.md).

## Entorno observado

- Acceso exclusivamente por <https://hpc.cedia.edu.ec> y Open OnDemand.
- Apptainer 1.3.0; Slurm 22.05.5.
- `$HOME/pytorch_24.01-py3.sif`, 10135441408 bytes, SHA-256
  `b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3`.
- NGC PyTorch 24.01: Ubuntu 22.04, Python 3.10.12, torch 2.2.0a0,
  torchvision 0.17.0a0, CUDA 12.3 y cuDNN 8907.
- GPU validada en sesión interactiva: A100-SXM4-40GB, driver 535.247.01.
- Partición de producción `gpu`, máximo observado de dos días; nodos con ocho
  A100. La sesión interactiva de diagnóstico usó 32 CPU, 96 GiB y una A100.
- HOME NFS con 33 TB libres en el volumen; no apareció cuota individual.

No se presupone Conda remoto. Los paquetes ausentes observados fueron
scikit-image, timm, transformers y segmentation-models-pytorch; el código no
requiere scikit-image para las métricas actuales. El bootstrap instala los otros
tres y `gdown` en un venv persistente sin reemplazar el stack CUDA.

## Limitación del runtime

El host no ofreció `squashfuse`, `fuse2fs` ni `gocryptfs`, por lo que Apptainer
convierte la SIF a un sandbox temporal en cada `exec`. No se requieren
privilegios ni una imagen nueva: `run_in_container.sh` agrupa una etapa completa
en un único `exec`. El bootstrap instala todo en una invocación; cada job Slurm
usa una invocación para su entrenamiento/evaluación.

## Paralelismo comprobado en código

Darknet se ejecuta sin `-gpus`; una tarea usa la única GPU que Slurm expone.
B2 no implementa DDP ni DataParallel. YOLO usa array 0–4 para folds 0–4; B2 usa
array 0–74, con `task // 5` para S01–S15 y `task % 5` para folds 0–4. Los
adaptadores aceptan `--device cuda`; el wrapper sustituye el prefijo Conda del
catálogo por el Python persistente mediante `THESIS_ADAPTER_PYTHON`.

Recursos iniciales por tarea:

| Job | CPU | RAM | GPU | Tiempo | Motivo |
|---|---:|---:|---:|---:|---|
| smoke | 4 | 24 GiB | 1 A100 | 1 h | una imagen/AViT |
| YOLO fold | 16 | 48 GiB | 1 A100 | 24 h | entrenamiento Darknet CUDA |
| B2 modelo-fold | 8 | 48 GiB | 1 A100 | 24 h | batch efectivo actual 1 |
| A/B1/C0–C3 | 8 | 48 GiB | 1 A100 | 48 h | 16 backends secuenciales |

La inferencia OpenCV DNN de P0 permanece en CPU porque la compilación CUDA de
OpenCV no fue confirmada; la GPU del job sigue reservada para los backends
neuronales. Son estimaciones iniciales, no mediciones de utilización. Ajuste futuras
reservas con `sacct MaxRSS`, tiempos y VRAM reales sin cambiar parámetros
científicos silenciosamente.

## Dependencias por familia

| Familia | Dependencias adicionales al SIF | Estado CEDIA |
|---|---|---|
| S01, S02, S05–S06, S10, S14, S15 | timm, einops | instalable en `.cedia/venv`; pendiente ejecutar smoke |
| S03, S11–S13 | stack PyTorch/Pillow de la SIF | pendiente cargar checkpoint en A100 |
| S04, S07–S08 | segmentation-models-pytorch 0.5.0, timm | fijado en requirements; pendiente preflight CEDIA |
| S09 | transformers 4.40.2 | fijado; pendiente cargar checkpoint CEDIA |
| S16/P0/métricas | OpenCV, NumPy, SciPy, Pillow, PyYAML | presentes; validable CPU |
| Darknet | GCC/G++, make, NVCC 12.3, cuDNN | presentes; compilación A100 pendiente |

Una incompatibilidad real puede justificar venvs por backend, pero no se crea
otra SIF ni quince entornos especulativos. `B2_VENV_ROOT` se conserva como vía de
escape documentada en la plantilla.

## Transporte sin SSH

Opciones compatibles: clonar/pull desde GitHub en la terminal web; descargar
fuentes oficiales desde CEDIA; subir archivos mediante Open OnDemand Files; o
reutilizar datasets ya presentes después de verificar manifests/hashes. No se
requiere ni documenta SSH, scp, sftp, rsync o túneles.
