# Benchmark reproducible S01–S16

Este documento describe la implementación posterior a la Fase 0. No modifica ni
reinterpreta su handoff. Los resultados de prueba, parciales o sintéticos no son
resultados científicos.

## Estado real

- S01–S15 son exactamente las quince entradas, en orden de `priority`, de
  `configs/segmentation_models.json`; conservan los subprocesos y el entorno
  `thesis-avit`.
- S16 es un único backend GrabCut con modos `classic` y `robust`.
- A, B1, B2 y C0–C3 tienen identidades distintas en manifests. B2 exige pesos
  con metadatos `protocol=B2`; nunca acepta pesos nativos bajo otra etiqueta.
- P0 implementa FOV, DullRazor/inpainting, YOLOv3 o su estado explícito de
  ausencia/fallo, ROI y transformación de coordenadas.
- Los recursos pesados son locales e ignorados por Git. Las rutas de trabajo
  son `data/raw/isic2018_task1`, `data/raw/imapp`,
  `data/raw/isic2018_novice_masks` y `models/yolov3-darknet`; su presencia debe
  verificarse con manifests, no inferirse de Git. B2 continúa bloqueado hasta
  que existan sus 75 checkpoints reales.

### Inventario local verificado (2026-08-09)

- ISIC 2018 Task 1: 2.594 imágenes/máscaras de training, 100 de validation
  con máscara y 1.000 imágenes de test. El ground truth de test permanece
  comprimido en `data/raw/isic2018_task1/sealed/archives/` y no fue extraído.
- IMA++ v1.1: 2.394 imágenes, 22.472 segmentaciones y particiones oficiales
  1.675/240/479; sus MD5 de Zenodo y los SHA-256 locales fueron verificados.
- ISIC 2018 Novice: 11.720 pares imagen/máscara, registrados y deshabilitados
  para selección o entrenamiento principal.
- Fitzpatrick: 93 imágenes, sin solapamientos con ISIC 2018, IMA++ o Novice;
  las 93 continúan `pending_annotation` y no tienen métricas de exactitud.
- Darknet original: commit `f86901f6177dfc6116360a13cc06ab680e0c86b0`,
  binario CPU SHA-256 `b4817514584d850387e6accabc88f8a7f64d2424f218420c83101d99195d7a4e`
  y `darknet53.conv.74` SHA-256
  `2495c2690283e4e0bc2050cbd4660b77a8074e14e9c11150c6412fd63db496a7`.
- Los cinco folds YOLO están preparados localmente en
  `results/benchmark_v1/yolo/`, cada uno con 2.594 etiquetas y auditoría sin
  fallos. Sus listas contienen rutas de la laptop y deben regenerarse dentro
  de CEDIA. Los cinco entrenamientos continúan pendientes.
- Hay 0/75 checkpoints B2. B2 y el test sellado permanecen bloqueados. Las
  plantillas remotas versionables están bajo `scripts/hpc/`; los antiguos
  command files bajo `results/` son artefactos locales y no deben enviarse.

## Datasets

ISIC 2018 Task 1 se descarga desde la página oficial del reto. El descargador
reanuda los seis ZIP oficiales, publica cada archivo atómicamente y conserva el
ground truth de test comprimido bajo `sealed/`; no lo extrae ni lo usa:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py download-isic2018 \
  --output "data/raw/isic2018_task1" --confirm-download
```

Después de extraer únicamente Training, Validation y las imágenes de Test:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py extract-isic2018 \
  --data-root "data/raw/isic2018_task1"
```

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py import-isic2018 \
  --data-root "data/raw/isic2018_task1" --split train \
  --metadata-root "data/raw/isic2018_task1/metadata" \
  --output "data/raw/isic2018_task1/manifests/isic2018_task1_train.json"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py verify \
  --manifest "data/raw/isic2018_task1/manifests/isic2018_task1_train.json" \
  --data-root "data/raw/isic2018_task1"
```

Repita `import-isic2018` con `validation` y `test`. El importador decodifica
imágenes/máscaras, comprueba dimensiones, cuenta 2.594/100/1.000 y guarda
SHA-256. No utiliza ISIC 2017.

IMA++ v1.1 se verifica contra los MD5 publicados por Zenodo. Revise primero la
licencia CC BY-NC-ND 4.0:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py download-imapp-metadata \
  --output "data/raw/imapp/source" --include-segmentations --confirm-download

conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py import-imapp \
  --source-root "data/raw/imapp/source" --data-root "data/raw/imapp" \
  --output "data/raw/imapp/manifests"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py download-isic-images \
  --manifest "data/raw/imapp/manifests/imapp_train.json" \
  --output "data/raw/imapp/images" --metadata-output "data/raw/imapp/image_metadata_api" \
  --confirm-download
```

Repita `download-isic-images` para `imapp_validation.json` e `imapp_test.json`
y vuelva a ejecutar `import-imapp` para congelar SHA-256, tamaños y dimensiones.
Las imágenes se importan desde la colección ISIC autorizada, no se inventan.
El módulo incluye majority voting y STAPLE reproducible para el subconjunto
multi-anotador; se reportan además máscaras individuales.

La auditoría local de IMA++ v1.1 detecta identidades compartidas entre sus
particiones oficiales (72 colisiones por `patient_id` y 40 por `lesion_id`).
Se conservan las particiones publicadas para evaluación externa, pero no deben
usarse como folds independientes de entrenamiento sin reagrupar primero.

Las Novice Masks se registran deshabilitadas:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py register-novice \
  --masks-root "data/raw/isic2018_novice_masks/Masks" \
  --data-root "data/raw/isic2018_novice_masks" \
  --mapping "data/raw/isic2018_novice_masks/supplements/Masks_to_IsicId_mapping.csv" \
  --metadata "data/raw/isic2018_novice_masks/metadata.csv" \
  --output "data/raw/isic2018_novice_masks/manifest.json"
```

Su manifest contiene `NO APTO PARA ELEGIR EL MODELO PRINCIPAL` y el motor no
permite seleccionarlas por defecto.

Para registrar la cohorte Fitzpatrick local y detectar duplicados por SHA-256:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py register-fitzpatrick \
  --images-root "data/raw/isic_segmentation_pilot_images" \
  --metadata "data/raw/isic_fitzpatrick_metadata_full.csv" --data-root "data/raw" \
  --output "data/interim/fitzpatrick_external_manifest.json"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/manage_datasets.py audit-overlap \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_train.json" \
  --manifest "data/interim/fitzpatrick_external_manifest.json" \
  --output "reports/fitzpatrick_overlap_audit.json"
```

## Folds y P0

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/build_folds.py \
  --dataset isic2018_task1 --split train --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_train.json" \
  --folds 5 --seed 20260806 --output "/RUTA/ISIC2018/manifests/folds.json"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/prepare_p0.py \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_train.json" --data-root "/RUTA/ISIC2018" \
  --artifact-root "results/benchmark_v1/preprocessing" --confirm-run
```

Los folds agrupan `patient_id`, `lesion_id` y `duplicate_group_id`; si todos
faltan, el fallback declarado es `image_id`. P0 conserva la imagen original y
su caché se invalida por hash de imagen, configuración y detector.

## YOLOv3/Darknet-53

Use la copia original fijada de Darknet y el `yolov3.cfg` oficial. No se
sustituye por YOLOv5/8/11. En una máquina sin CUDA ni `opencv.pc`, el instalador
compila explícitamente `GPU=0 CUDNN=0 OPENCV=0` y registra esa decisión.

```fish
conda run --no-capture-output -n tesis-sam python scripts/setup_yolov3_darknet.py \
  --revision "f86901f6177dfc6116360a13cc06ab680e0c86b0" \
  --download-bootstrap --confirm-download --build

conda run --no-capture-output -n tesis-sam python scripts/benchmark/yolov3.py prepare-fold \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_train.json" --data-root "/RUTA/ISIC2018" \
  --folds "/RUTA/ISIC2018/manifests/folds.json" --fold 0 \
  --output "results/benchmark_v1/yolo/fold-0" --seed 20260806

conda run --no-capture-output -n tesis-sam python scripts/benchmark/yolov3.py configure \
  --source-cfg "/RUTA/DARKNET/cfg/yolov3.cfg" \
  --output "results/benchmark_v1/yolo/fold-0/lesion-yolov3.cfg"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/yolov3.py train \
  --darknet "/RUTA/DARKNET/darknet" --data "results/benchmark_v1/yolo/fold-0/lesion.data" \
  --cfg "results/benchmark_v1/yolo/fold-0/lesion-yolov3.cfg" \
  --initial-weights "/RUTA/DARKNET/darknet53.conv.74" \
  --output "results/benchmark_v1/yolo/fold-0/training" --confirm-training
```

`--resume /RUTA/A/BACKUP.weights` reanuda. La selección de confianza, NMS y
margen se hace solamente con validación; después:

La ejecución en CEDIA se prepara exclusivamente mediante Open OnDemand. No se
presupone Conda remoto ni una copia local de la SIF. Ejecute primero el
diagnóstico y siga [`CEDIA_OPEN_ONDEMAND.md`](CEDIA_OPEN_ONDEMAND.md). La
plantilla versionable es `scripts/hpc/train_yolo_cedia.slurm`; usa cinco tareas,
dos concurrentes y una A100 por tarea. Las rutas del proyecto, datos y SIF son
variables obligatorias observadas en CEDIA, no valores inventados.

Antes de enviarlo en un nodo GPU, compile la misma revisión de Darknet allí
con `GPU=1`, `CUDNN=1` y los flags admitidos por ese nodo. El binario local
registrado fue compilado para CPU y no convierte en GPU un job por solicitar
`--gres=gpu:1`. Si la cola o módulos de CEDIA difieren, cambie solo las
directivas de recursos y el paso de activación del entorno, no los folds,
semillas, CFG ni pesos iniciales.

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/yolov3.py infer-validation \
  --cfg "results/benchmark_v1/yolo/fold-0/lesion-yolov3.cfg" \
  --weights "/RUTA/lesion-yolov3_best.weights" \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_train.json" --data-root "/RUTA/ISIC2018" \
  --folds "/RUTA/ISIC2018/manifests/folds.json" --fold 0 \
  --output "results/benchmark_v1/yolo/fold-0/raw_validation.json"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/yolov3.py validate \
  --predictions "results/benchmark_v1/yolo/fold-0/raw_validation.json" \
  --output "results/benchmark_v1/yolo/fold-0/validation.json"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/yolov3.py freeze \
  --cfg "results/benchmark_v1/yolo/fold-0/lesion-yolov3.cfg" \
  --weights "/RUTA/lesion-yolov3_best.weights" --validation-report "/RUTA/validation.json" \
  --confidence 0.25 --nms 0.45 --margin 0.2 \
  --output "results/benchmark_v1/yolo/fold-0/frozen.json"
```

Asigne la ruta de ese JSON a `p0.yolo.frozen_manifest` en una configuración
local derivada de `configs/benchmark/default.json`; P0 verificará los hashes y
usará directamente confianza, NMS y margen congelados. El fallback FOV queda marcado como
`unavailable`, `configured_error` o `configured_no_detection`; nunca se
presenta como una detección exitosa.

## Evaluación y ablaciones

Una imagen y cinco imágenes:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/run_evaluation.py \
  --evaluation A --dataset isic2018_task1 --split validation --models all \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_validation.json" \
  --data-root "/RUTA/ISIC2018" --limit 1 --confirm-run

conda run --no-capture-output -n tesis-sam python scripts/benchmark/run_evaluation.py \
  --evaluation B1 --dataset isic2018_task1 --split validation --models all \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_validation.json" \
  --data-root "/RUTA/ISIC2018" --limit 5 --confirm-run
```

Para benchmarking temporal use `--warmup 1 --repetitions 3`. Se registran
mediana, P25, P75, P95, CPU y RAM de proceso. El tiempo del adaptador incluye
arranque/carga del subproceso y se etiqueta así; `end_to_end_time_ms` incluye P0
y posprocesado por separado.

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/run_ablation.py \
  --conditions C0,C1,C2,C3 --dataset isic2018_task1 --split validation --models all \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_validation.json" \
  --data-root "/RUTA/ISIC2018" --confirm-run
```

## B2 reanudable y HPC

El entrenador no acepta imágenes crudas como sustituto de P0. Para revisar los
75 jobs sin iniciarlos:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/train_b2.py \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_train.json" --data-root "/RUTA/ISIC2018" \
  --folds "/RUTA/ISIC2018/manifests/folds.json" --p0-root "results/benchmark_v1/preprocessing" \
  --output "results/benchmark_v1/b2" --fold all --models all --epochs 100 --dry-run \
  --command-file "results/benchmark_v1/b2/commands.txt"
```

Quite `--dry-run`, añada `--confirm-training` y opcionalmente `--device cuda`.
Una interrupción se recupera repitiendo con `--resume --confirm-training`.
El command file generado localmente contiene rutas de la laptop y no es una
plantilla remota. Para CEDIA use `scripts/hpc/train_b2_cedia.slurm`, después del
diagnóstico y únicamente cuando los cinco detectores estén validados y
congelados. La plantilla mapea 75 tareas a S01–S15 × folds 0–4, limita la
concurrencia a dos y solicita una A100 por tarea. Valida hashes YOLO y
`split=train` antes de iniciar; no presupone que `thesis-avit` exista en CEDIA.

B2 se evalúa solo con un registro que incluya los quince paths y SHA-256:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/freeze_b2.py \
  --training-root "results/benchmark_v1/b2" --base-config "configs/benchmark/default.json" \
  --output "results/benchmark_v1/b2_frozen.json" --note "Protocolo validado y congelado"

conda run --no-capture-output -n tesis-sam python scripts/benchmark/run_evaluation.py \
  --evaluation B2 --b2-config "results/benchmark_v1/b2_frozen.json" \
  --dataset isic2018_task1 --split validation --models all \
  --manifest "/RUTA/ISIC2018/manifests/isic2018_task1_validation.json" \
  --data-root "/RUTA/ISIC2018" --confirm-run
```

## Anotación y web

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/annotations.py init \
  --manifest "data/interim/fitzpatrick_external_manifest.json" \
  --project "data/interim/fitzpatrick_annotations_v1"

conda run --no-capture-output -n tesis-sam python scripts/serve_segmentation_review.py \
  --host 127.0.0.1 --port 8000
```

Abra <http://127.0.0.1:8000>. La interfaz conserva el revisor nativo, lee runs
reales de `results/benchmark_v1`, muestra las etapas disponibles y ofrece
primer lector, segundo lector y adjudicación. Exporte el consenso con:

```fish
conda run --no-capture-output -n tesis-sam python scripts/benchmark/annotations.py export \
  --project "data/interim/fitzpatrick_annotations_v1" \
  --output "data/interim/fitzpatrick_external_adjudicated.json"
```

## Test sellado

No prepare ni ejecute hasta congelar B2. `sealed_test.py prepare` exige árbol
Git limpio, 1.000 imágenes, hashes de configuración/manifiesto y quince pesos
B2. `audit` detecta cambios. `execute` exige literalmente
`EJECUTAR TEST SELLADO UNA VEZ`, registra todo intento y no permite `--limit`,
IDs manuales ni métricas intermedias. Al terminar cambia a `sealed` o `failed` y
no permite una segunda ejecución.

## Validación

```fish
conda run --no-capture-output -n tesis-sam python -m unittest discover -s tests -v
conda run --no-capture-output -n tesis-sam python -m py_compile scripts/*.py scripts/benchmark/*.py scripts/adapters/*.py src/thesis_fitzpatrick/*.py
node --check web/app.js
python -m json.tool configs/benchmark/default.json >/dev/null
python -m json.tool configs/benchmark/datasets.json >/dev/null
git diff --check
```

Matriz: implementación de datasets, P0, S16, A/B1/B2, C0–C3, métricas,
estadística, anotación, web y sellado está disponible; la evidencia científica
queda pendiente de datos oficiales, entrenamiento YOLO, anotación humana,
entrenamiento B2 y cómputo de los experimentos completos.
