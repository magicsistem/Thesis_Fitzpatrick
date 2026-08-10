# Ejecución en CEDIA mediante Open OnDemand

Esta guía prepara la ejecución remota sin asumir acceso por línea de comandos
desde la laptop. No certifica todavía que `pytorch_24.01-py3.sif` sea compatible:
esa conclusión depende del diagnóstico ejecutado dentro de CEDIA.

## Evidencia y datos pendientes

Información histórica proporcionada por el usuario, pendiente de confirmar en
el portal actual:

- portal: <https://hpc.cedia.edu.ec>;
- HOME observado: `/home/miguel.benavides__yachaytech.edu.ec`;
- partición `gpu` y GPU NVIDIA A100 SXM4 de 40 GB;
- imagen denominada `pytorch_24.01-py3.sif`, con Python 3.10.12 observado;
- reserva anterior: 64 CPU, 120 GB RAM y dos A100 durante 24 horas.

La ubicación actual de la SIF, el proyecto, los datasets y el almacenamiento de
resultados no se conocen. Tampoco están confirmados el runtime de contenedores,
cuotas, módulos, QoS ni paquetes dentro de la imagen. Los entornos locales
`tesis-sam` y `thesis-avit` no se presuponen en CEDIA.

## Diagnóstico de solo lectura

Hay dos maneras de ejecutarlo en Open OnDemand:

1. Abra la aplicación que el portal muestre como terminal del clúster (en
   instalaciones habituales aparece bajo `Clusters → Shell Access`, pero el
   nombre puede haber cambiado), sitúese en la copia remota del repositorio y
   ejecute el comando siguiente.
2. Abra la aplicación interactiva con GPU que ofrezca el portal. Como punto de
   partida histórico, solicite partición `gpu`, 16 horas, 32 CPU, 96 GB RAM y
   una A100 de 40 GB. Desde la terminal de esa sesión ejecute el mismo comando;
   esta modalidad permite comprobar la GPU visible.

```bash
cd "$HOME/Thesis_Fitzpatrick"  # solo si esta es la ubicación que usted eligió
bash scripts/hpc/inspect_cedia_environment.sh 2>&1 | \
  tee "cedia_environment_$(date +%Y%m%d_%H%M%S).txt"
```

Si la búsqueda limitada bajo `$HOME` no encuentra la imagen, repita con la ruta
que usted observe en el administrador de archivos o terminal del portal:

```bash
SIF_PATH="/ruta/observada/en/CEDIA/pytorch_24.01-py3.sif" \
bash scripts/hpc/inspect_cedia_environment.sh 2>&1 | \
  tee "cedia_environment_with_sif_$(date +%Y%m%d_%H%M%S).txt"
```

Puede añadir directorios accesibles y concretos, sin recorrer todo el sistema:

```bash
SIF_SEARCH_ROOTS="/primera/ruta:/segunda/ruta" \
bash scripts/hpc/inspect_cedia_environment.sh
```

Devuelva el archivo `cedia_environment_*.txt` o
`cedia_environment_with_sif_*.txt`. Si hizo ambos diagnósticos, devuelva ambos.
No hace falta enviar la SIF.

## Matriz provisional de dependencias

La columna final permanece deliberadamente sin resolver hasta leer el informe
de CEDIA.

| Métodos/componente | Requisitos observados en código o archivos locales | Estrategia persistente posible | SIF |
|---|---|---|---|
| S01 AViT | PyTorch, torchvision, NumPy, Pillow, timm, einops, PyYAML; el repositorio upstream declara además fvcore/monai para otros flujos | venv común si las versiones cargan el checkpoint | pendiente de diagnóstico CEDIA |
| S02 UltraLight-VM-UNet | PyTorch, NumPy, Pillow, timm, einops; el adaptador sustituye `mamba_ssm` por compatibilidad local | venv común; no compilar Mamba salvo fallo demostrado | pendiente de diagnóstico CEDIA |
| S03 BA-Transformer | PyTorch, torchvision, NumPy, Pillow; el adaptador evita el peso ResNet absoluto del autor | venv común si el código upstream carga | pendiente de diagnóstico CEDIA |
| S04 | PyTorch, torchvision y `segmentation-models-pytorch==0.5.0` | venv para SMP si hay conflicto | pendiente de diagnóstico CEDIA |
| S05–S06 SkinMamba | PyTorch, timm, einops, NumPy, Pillow; compatibilidad de selective scan incluida por el adaptador | venv común primero; extensión CUDA solo si el diagnóstico y una prueba muestran que es necesaria | pendiente de diagnóstico CEDIA |
| S07–S08 | PyTorch, torchvision y `segmentation-models-pytorch==0.5.0` | mismo venv SMP de S04 | pendiente de diagnóstico CEDIA |
| S09 SegFormer | PyTorch, torchvision, transformers, NumPy, Pillow | venv separado si transformers entra en conflicto | pendiente de diagnóstico CEDIA |
| S10 y S14 VM-UNet | PyTorch, timm, einops, NumPy, Pillow; selective scan de compatibilidad local | venv común primero | pendiente de diagnóstico CEDIA |
| S11 | PyTorch, NumPy y Pillow; arquitectura Attention U-Net local | Python del contenedor o venv común | pendiente de diagnóstico CEDIA |
| S12–S13 | PyTorch, NumPy y Pillow; arquitecturas locales U-Net/Inception | Python del contenedor o venv común | pendiente de diagnóstico CEDIA |
| S15 DeLightSAM | PyTorch, torchvision, timm, einops, NumPy y Pillow; código fuente externo fijado | venv propio si sus versiones resultan incompatibles | pendiente de diagnóstico CEDIA |
| S16 GrabCut/P0/métricas | Python 3.10, NumPy, OpenCV, Pillow, SciPy y scikit-image; PyYAML para configuración | venv de benchmark persistente si faltan paquetes | pendiente de diagnóstico CEDIA |
| B2 | Lo anterior según backend; AdamW y CUDA de PyTorch | `B2_VENV_ROOT/S01`…`S15` permite separar solo los métodos que realmente lo necesiten | pendiente de diagnóstico CEDIA |
| Darknet YOLOv3 | gcc, g++, make, CUDA/nvcc y cuDNN para una compilación GPU; CFG y bootstrap weights ya fijados | binario persistente bajo el proyecto, compilado una vez después del diagnóstico | pendiente de diagnóstico CEDIA |

No se debe crear otra imagen antes de demostrar una incompatibilidad. Si faltan
paquetes, la primera opción es un venv persistente creado desde el Python del
contenedor; las plantillas aceptan `VENV_PATH` o, para B2, venvs por método bajo
`B2_VENV_ROOT`. La instalación se hará una sola vez fuera de los jobs, después
de revisar el diagnóstico. El proyecto, datos y resultados se enlazan al
contenedor mediante `--bind` y la GPU mediante `--nv`.

## Paralelismo comprobado en el código

- Darknet se invoca sin `-gpus`; cada proceso usa la única GPU que SLURM le
  haga visible. No hay entrenamiento multi-GPU implementado.
- B2 no usa DDP ni `DataParallel`. Cada tarea carga un modelo y lo mueve a
  `cuda`; por tanto usa una GPU.
- `CUDA_VISIBLE_DEVICES` lo establece SLURM. Las plantillas lo registran y no
  lo sobrescriben.
- YOLO mapea `SLURM_ARRAY_TASK_ID=0..4` directamente al fold 0..4.
- B2 mapea `task // 5` a S01..S15 y `task % 5` al fold 0..4.
- Darknet reanuda desde `RESUME_WEIGHTS` o el backup más reciente. B2 añade
  `--resume` cuando existe `training_state.pt`.
- Los checkpoints B2 se congelan después con `freeze_b2.py`; este exige 75
  checkpoints y verifica protocolo, modelo, fold y SHA-256.
- Las plantillas rechazan manifests distintos de `split=train`. B2 además
  verifica los cinco YOLO congelados y sus hashes antes de arrancar.

La solicitud inicial por tarea es una A100 de 40 GB, 32 CPU y 60 GB RAM,
obtenida conservadoramente al dividir la reserva histórica de dos GPU, 64 CPU y
120 GB. No implica que el entrenamiento use eficazmente todos esos CPU o RAM.
Tras medir un job representativo, reduzca la solicitud si el uso real lo permite.

## Plantillas SLURM

Las fuentes versionables son:

- `scripts/hpc/train_yolo_cedia.slurm`: cinco folds,
  `--array=0-4%2`, una GPU por tarea;
- `scripts/hpc/train_b2_cedia.slurm`: quince modelos por cinco folds,
  `--array=0-74%2`, una GPU por tarea.

No las envíe hasta validar el diagnóstico, preparar los folds en las rutas
remotas y confirmar una compilación GPU de Darknet. Ejemplo de variables; solo
`$HOME` es conocido y las demás rutas deben coincidir con lo observado:

```bash
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="/ruta/confirmada/en/CEDIA/pytorch_24.01-py3.sif"
export DARKNET_GPU_CONFIRMED=YES
# Envío manual posterior, no durante el diagnóstico:
# sbatch scripts/hpc/train_yolo_cedia.slurm
```

B2 permanece bloqueado hasta validar y congelar los cinco detectores:

```bash
export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="/ruta/confirmada/en/CEDIA/pytorch_24.01-py3.sif"
export YOLO_FROZEN_ROOT="$PROJECT_ROOT/results/benchmark_v1/yolo"
# Envío manual posterior:
# sbatch scripts/hpc/train_b2_cedia.slurm
```

Los scripts registran job, tarea, host, fecha UTC, GPU visible y Git HEAD en
`results/benchmark_v1/{yolo,b2}/logs/`. Los checkpoints y estados se escriben
en rutas persistentes. Una señal conserva el último backup Darknet o la última
época B2 terminada para volver a presentar la misma tarea.

## Transferencia compatible con Open OnDemand

Las opciones admitidas son:

1. Después del commit y push manuales, clonar o actualizar GitHub desde la
   terminal web de CEDIA, si el nodo permite salida a Internet.
2. Subir un archivo comprimido del repositorio con el administrador de archivos
   del portal y extraerlo desde la terminal web.
3. Descargar datasets desde sus fuentes oficiales usando la terminal web, si
   CEDIA permite salida; los descargadores del proyecto son reanudables.
4. Subir datasets mediante el administrador web cuando su tamaño lo permita.
5. Reutilizar copias ya presentes en CEDIA después de verificar manifests y
   SHA-256.

La elección depende de cuota, almacenamiento y conectividad revelados por el
diagnóstico. El test sellado no se transfiere, inspecciona ni ejecuta durante
esta preparación.

## Orden posterior

1. Ejecutar el diagnóstico en terminal normal y, si es posible, en una sesión
   interactiva con una A100.
2. Revisar runtime, ruta/hash de SIF, paquetes, CUDA/cuDNN, compiladores, cuota y
   límites SLURM.
3. Elegir y verificar las rutas persistentes de proyecto, datos y resultados.
4. Preparar una sola vez los venv estrictamente necesarios y compilar Darknet
   GPU dentro del entorno confirmado.
5. Regenerar en CEDIA las etiquetas/listas YOLO, porque contienen rutas
   absolutas del clúster; auditar round-trip y fugas.
6. Presentar los cinco folds YOLO, validar cada fold y congelar sus hashes.
7. Generar P0 con el detector correspondiente a cada fold.
8. Presentar B2; verificar y congelar los 75 checkpoints.
9. Ejecutar evaluación de desarrollo. El test sellado continúa cerrado hasta
   la congelación científica final.
