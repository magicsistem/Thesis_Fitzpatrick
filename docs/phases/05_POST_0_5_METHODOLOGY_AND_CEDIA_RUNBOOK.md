# Metodología posterior a Fase 0.5 y runbook CEDIA

Estado metodológico congelado para el piloto posterior al benchmark de segmentación.
Este documento **no reescribe ni borra** los resultados históricos A/B1 ni los cinco
folds YOLO existentes. Su objetivo es eliminar la ambigüedad que produjo el uso
provisional de un único fold para B1 y dejar una ruta reproducible hasta la
selección del piloto MSKCC.

## Decisiones que no deben volver a cambiar durante el piloto

1. Los cinco YOLO de cross-validation son modelos **CV/OOF**. Sirven para medir
   generalización y seleccionar hiperparámetros. Ninguno de esos cinco pesos se
   usa como YOLO definitivo de B1.
2. Después del CV se entrena **un solo YOLO final-refit** sobre todo el
   `isic2018_task1_train_disjoint`, manteniendo fuera el
   `isic2018_task1_validation_disjoint` y el test sellado.
3. El final-refit no reoptimiza confidence, NMS ni margen. Esos valores se heredan
   de los cinco folds congelados. En el estado actual los cinco folds deben tener
   consenso; el código se detiene si no lo tienen en lugar de escoger un fold.
4. B1 definitivo, ablaciones y MSKCC utilizan únicamente `model_role=final_refit`.
5. Las cinco ablaciones son one-component-out y quedan fijadas antes de MSKCC:
   `B1_FULL`, `B1_NO_HAIR`, `B1_NO_YOLO`, `B1_NO_FOV`, `B1_NO_POST`.
6. El TOP-3 se selecciona con el score predefinido de 100 puntos descrito abajo.
   El score y los umbrales de exclusión se congelan **antes** de observar MSKCC.
7. La muestra MSKCC intenta 10 imágenes por MST 1..10. Si eso no es factible bajo
   las restricciones de independencia, cambia automáticamente a 20 imágenes por
   cada par MST 1-2, 3-4, 5-6, 7-8, 9-10.
8. El objetivo de la Fase IX se llama **human-corrected reference mask**.
9. Las 20 referencias independientes parten de una máscara candidata de un SAM de
   la familia ya investigada y terminan en corrección humana completa; el SAM no
   se considera ground truth automático.
10. Fairness primario: Dice vs MST y Dice vs ITA. Secundario: Dice vs FST y Dice
    vs L*. También se auditan fallback de YOLO y predicciones degeneradas por tono.
11. La clasificación benigno/maligno/desconocido y cualquier algoritmo de
    corrección de bias ocurren **después** de esta metodología de segmentación y
    no pueden retroalimentar la selección del TOP-3.

---

# Por qué existen cinco pesos de CV y un peso final

En 5-fold CV cada modelo aprende con cuatro quintos del conjunto de entrenamiento
y predice el quinto restante. Los cinco checkpoints son necesarios para obtener
predicciones out-of-fold sin fuga. Esos pesos no son cinco candidatos entre los
que se deba elegir "el mejor".

Cuando el CV termina y los hiperparámetros quedan fijados, se realiza un **refit**:

```text
train_disjoint
   ├── fold-0 train/val → checkpoint CV 0
   ├── fold-1 train/val → checkpoint CV 1
   ├── fold-2 train/val → checkpoint CV 2
   ├── fold-3 train/val → checkpoint CV 3
   └── fold-4 train/val → checkpoint CV 4
                 │
                 └── hiperparámetros congelados
                              │
                              ▼
                 TODO train_disjoint
                              │
                              ▼
                    YOLO FINAL-REFIT
                              │
                              ▼
                  validation_disjoint
```

`validation_disjoint` sigue siendo evaluación de desarrollo y no entra al refit.
El test sellado tampoco entra.

El número final de batches conserva aproximadamente la exposición por imagen del
CV:

```text
final_max_batches = round(
    cv_max_batches * N_final / mean(N_train_de_cada_fold)
)
```

Con un 5-fold aproximadamente 80/20, esto suele ser cercano a 1.25× los batches
de un fold. Se calcula desde los manifests reales, no se fija a mano.

---

# Fase 1 — YOLO final-refit

## 1.1 Prerrequisitos CEDIA

En el login node:

```bash
cd "$HOME/Thesis_Fitzpatrick"
git status --short
git rev-parse HEAD

export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"
export CV_YOLO_ROOT="$PROJECT_ROOT/results/benchmark_v1/yolo"
export FINAL_YOLO_ROOT="$CV_YOLO_ROOT/final"
```

Verifique primero los cinco folds históricos:

```bash
scripts/hpc/run_in_container.sh -- \
  python scripts/benchmark/yolov3.py validate-folds \
  --root "$CV_YOLO_ROOT"
```

Debe reportar cinco folds válidos.

## 1.2 Preparación

```bash
bash scripts/hpc/post05_prepare_yolo_final_cedia.sh
```

Se crea:

```text
results/benchmark_v1/yolo/final/
├── images/
├── labels/
├── backup/
├── training/
├── lesion-yolov3-final.cfg
├── lesion.data
├── lesion.names
├── train.txt
└── prepare_manifest.json
```

`valid=train.txt` existe únicamente porque Darknet espera el campo `valid` en
`lesion.data`. El entrenamiento final se lanza **sin `-map`** y ninguna métrica
de ese alias se usa para selección.

## 1.3 Entrenamiento

Confirme que la compilación GPU de Darknet ya pasó el preflight existente y:

```bash
export DARKNET_GPU_CONFIRMED=YES
sbatch scripts/hpc/post05_train_yolo_final_cedia.slurm
```

Si Slurm interrumpe el job, vuelva a enviar exactamente el mismo `sbatch`.
`yolo_final_refit.py` solo adopta un checkpoint del mismo directorio `backup/`
si el contrato de cfg/datos/bootstrap coincide. Si no existe un checkpoint válido,
no inventa uno.

## 1.4 Freeze

Cuando `training/training_state.json` diga `completed`:

```bash
bash scripts/hpc/post05_freeze_yolo_final_cedia.sh
```

Esto hace dos cosas:

1. crea `results/benchmark_v1/yolo/final/frozen.json` con
   `model_role=final_refit` y `fold=null`;
2. crea `.cedia/post05/benchmark.final_b1.json` con cfg/weights/thresholds finales.

El freeze se detiene si confidence/NMS/margin no son idénticos en los cinco folds
CV. No escoge el fold con mejor resultado.

---

# Fase 2 — B1 definitivo

```bash
export DEVELOPMENT_MANIFEST="$DATA_ROOT/manifests/isic2018_task1_validation_disjoint.json"
export RUN_ID="post05-b1-final-$(date -u +%Y%m%dT%H%M%SZ)"
sbatch --export=ALL,RUN_ID="$RUN_ID" scripts/hpc/post05_run_b1_final_cedia.slurm
```

Anote `RUN_ID`; se necesita para el TOP-3.

El pipeline es:

```text
FOV
 ↓
hair detection + Telea
 ↓
YOLO FINAL-REFIT
 ↓
ROI
 ↓
S01 ... S16
 ↓
common postprocessing
```

Una no-detección de YOLO sigue usando el fallback documentado y se conserva como
resultado. Un `technical_failure` requiere reparación; una predicción degenerada
permanece en el denominador.

---

# Fase 3 — Ablaciones one-component-out

Las cinco condiciones son:

| Condición | FOV | Hair/inpainting | YOLO | Postprocesamiento |
|---|---:|---:|---:|---:|
| B1_FULL | Sí | Sí | Sí | Sí |
| B1_NO_HAIR | Sí | No | Sí | Sí |
| B1_NO_YOLO | Sí | Sí | No | Sí |
| B1_NO_FOV | No | Sí | Sí | Sí |
| B1_NO_POST | Sí | Sí | Sí | No |

Se ejecutan como array para no meter cinco benchmarks completos en un único job:

```bash
export RUN_PREFIX="post05-ablation-$(date -u +%Y%m%dT%H%M%SZ)"
sbatch --export=ALL,RUN_PREFIX="$RUN_PREFIX" scripts/hpc/post05_run_ablations_cedia.slurm
```

Los cinco runs serán:

```text
${RUN_PREFIX}-b1_full
${RUN_PREFIX}-b1_no_hair
${RUN_PREFIX}-b1_no_yolo
${RUN_PREFIX}-b1_no_fov
${RUN_PREFIX}-b1_no_post
```

Anote `RUN_PREFIX`.

---

# Fase 4 — Selección TOP-3

## Score pre-registrado

Total: 100 puntos.

### Región — 35

```text
17.5 × mean(Dice)
17.5 × mean(Jaccard)
```

### Borde — 20

```text
12 × mean(Boundary-F1)
8 × 1/(1 + mean(HD95_normalized))
```

### Estabilidad — 15

```text
15 × (1 - IQR(Dice))
```

### Fiabilidad — 15

```text
15 × clip(
  1 - technical_failure_rate - 0.75 × degenerate_prediction_rate,
  0,
  1
)
```

### Robustez P0 — 15

Primero se calcula la media de Dice para las cinco condiciones de ablación y:

```text
15 × clip(
  mean(Dice_condiciones) - std(Dice_condiciones),
  0,
  1
)
```

Esto evita premiar un método que sea bueno únicamente bajo una configuración
muy específica del P0. La calidad absoluta sigue aportando 55 puntos, por lo que
un modelo estable pero malo no puede ganar solo por estabilidad.

## Gates antes del ranking

Por defecto:

```text
technical_failure_rate <= 1%
degenerate_prediction_rate <= 10%
```

Un modelo que no pase el gate aparece en el reporte pero es inelegible para TOP-3.
No cambie estos límites después de mirar el ranking.

## Ejecutar selección y freeze

```bash
export B1_RUN_ID="<RUN_ID de Fase 2>"
export ABLATION_RUN_PREFIX="<RUN_PREFIX de Fase 3>"
bash scripts/hpc/post05_select_freeze_cedia.sh
```

Se generan:

```text
results/benchmark_v1/post05_selection/
├── segmenter_selection.json
├── segmenter_selection.csv
└── segmenter_selection.md

results/benchmark_v1/post05_scientific_freeze.json
```

El freeze exige un árbol Git limpio y registra hashes de YOLO, selección, B1,
ablaciones y commit.

**Este archivo debe existir antes de MSKCC.**

---

# Fase 5 — Adquisición y auditoría MSKCC

No se codifica una URL MSKCC fija: el endpoint o mecanismo de distribución puede
cambiar y puede requerir aceptar licencia/autenticación. El código obliga a que
la URL real quede explícita en el comando y se guarde en un audit.

Si existe un archivo descargable directo:

```bash
scripts/hpc/run_in_container.sh -- \
  python scripts/benchmark/prepare_mskcc_pilot.py download \
  --url '<URL_OFICIAL>' \
  --output data/raw/mskcc/<archivo> \
  --expected-sha256 '<SHA256 si está publicado>' \
  --confirm-download
```

Si es ZIP:

```bash
scripts/hpc/run_in_container.sh -- \
  python scripts/benchmark/prepare_mskcc_pilot.py extract-zip \
  --archive data/raw/mskcc/<archivo.zip> \
  --output data/raw/mskcc/extracted
```

Si la descarga requiere navegador/autenticación, coloque los archivos obtenidos
legalmente en CEDIA y continúe desde la construcción del registry. No se debe
saltar el registro de procedencia/checksum.

Variables normalizadas que intenta obtener el registry:

```text
image_id
patient_id
lesion_id
MST
FST
L*
a*
b*
ITA
acquisition_mode
anatomical_site
```

Los nombres de columnas comunes se detectan automáticamente. Para un nombre
particular use un mapa explícito, por ejemplo:

```bash
export MSKCC_COLUMN_MAP='{"image_id":"image_name","mst":"Monk Skin Tone"}'
```

---

# Fase 6 — Sample MSKCC de 100 imágenes

Con metadata e imágenes ya presentes:

```bash
export MSKCC_METADATA="/ruta/metadata.csv"
export MSKCC_IMAGES_ROOT="/ruta/images"
# opcional:
# export MSKCC_COLUMN_MAP='{"image_id":"image_name","mst":"Monk Skin Tone"}'

bash scripts/hpc/post05_mskcc_sample_cedia.sh
```

## Restricciones del selector

Por defecto:

- MST debe ser 1..10;
- la imagen debe existir;
- `lesion_id` debe existir;
- `patient_id` debe existir;
- una sola imagen por lesión;
- máximo dos lesiones por paciente;
- se usa una sola `acquisition_mode` cuando ese metadata existe;
- entre candidatos válidos se prioriza mayor completitud de FST, L*, b* e ITA;
- los empates usan SHA-256 con seed fijo, nunca el orden del CSV.

Estrategia primaria:

```text
MST 1  = 10
MST 2  = 10
...
MST 10 = 10
Total  = 100
```

Fallback automático solo si la primaria es imposible:

```text
MST 1-2  = 20
MST 3-4  = 20
MST 5-6  = 20
MST 7-8  = 20
MST 9-10 = 20
Total    = 100
```

Si tampoco es posible, el selector **falla** y escribe
`mskcc_sampling_failure.json`; no duplica lesiones ni excede silenciosamente el
límite por paciente.

Archivos esperados:

```text
results/mskcc_post05_pilot/
├── registry/
│   ├── mskcc_registry.json
│   ├── mskcc_registry.csv
│   └── mskcc_distribution_report.json
└── sample/
    ├── mskcc_sample_100.json
    ├── mskcc_sample_100.csv
    └── mskcc_sample_distribution.json
```

---

# Fases posteriores ya definidas, todavía no ejecutadas por este patch

## Fase 7 — TOP-3 sobre MSKCC

Ejecutar únicamente los tres métodos congelados, siempre detrás del P0 con
YOLO final-refit. Guardar las tres máscaras candidatas y todos los artefactos.

## Fase 8 — revisión ciega

Para 80 imágenes, presentar las tres alternativas como A/B/C en orden aleatorio.
El revisor selecciona la mejor y la corrige cuando sea necesario.

## Fase 9 — human-corrected reference mask

El producto final de revisión se denomina exactamente **human-corrected
reference mask**.

En las otras 20 imágenes, generar primero una referencia independiente con un
SAM previamente investigado como propuesta inicial y realizar corrección humana
completa **antes de revelar TOP1/TOP2/TOP3**. Estas 20 imágenes estiman el posible
anchoring bias del procedimiento de elección entre tres candidatos.

## Fase 10 — fairness de segmentación

Primarios:

```text
Dice vs MST
Dice vs ITA
```

Secundarios:

```text
Dice vs FST
Dice vs L*
```

Adicionales:

```text
YOLO fallback rate vs MST/ITA
segmenter degenerate rate vs MST/ITA
```

La inferencia de color de `clean_skin_mask` se compara con los valores de color
instrumental cuando estén disponibles. El análisis debe respetar clustering por
paciente.

## Después de Fase 10 — clasificación y bias correction

Solo después de aprobar el pipeline de segmentación se analiza el clasificador
benigno/maligno/desconocido. Primero se prueba si existe bias asociado al tono.
Si no existe una diferencia relevante, no se crea una corrección. Si existe, el
algoritmo corrector se desarrolla únicamente con development data, se congela y
se evalúa después en datos independientes.

---

# Checks antes de enviar trabajos largos

Después de aplicar este patch en la laptop y hacer push, en CEDIA ejecute antes
de cualquier entrenamiento:

```bash
cd "$HOME/Thesis_Fitzpatrick"
git pull --ff-only
git status --short
git rev-parse HEAD

export PROJECT_ROOT="$HOME/Thesis_Fitzpatrick"
export DATA_ROOT="$PROJECT_ROOT/data/raw/isic2018_task1"
export SIF_PATH="$HOME/pytorch_24.01-py3.sif"

scripts/hpc/run_in_container.sh -- python -m py_compile \
  src/thesis_fitzpatrick/post05.py \
  scripts/benchmark/yolo_final_refit.py \
  scripts/benchmark/run_component_ablation.py \
  scripts/benchmark/select_post05_models.py \
  scripts/benchmark/prepare_mskcc_pilot.py

scripts/hpc/run_in_container.sh -- pytest -q tests/test_post05.py
```

No continuar al job siguiente si cualquiera de esos comandos falla. Cada fase
produce un manifest/hash que funciona como gate de la siguiente.
