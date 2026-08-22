# Inspección visual post-Fase 0.5

## Contrato vigente

La interfaz es una auditoría interpretativa de solo lectura sobre el
`post05_scientific_freeze.json`. No reabre la selección, no cambia el TOP-3,
no recalcula el score, no modifica el YOLO final, no sobrescribe resultados y
no elimina imágenes. La fuente de verdad local se valida antes de habilitar la
revisión:

- TOP-3 congelado: `S01`, `S10`, `S14`.
- 64 imágenes de validación × 3 métodos = 192 casos.
- Cinco condiciones: `B1_FULL`, `B1_NO_HAIR`, `B1_NO_YOLO`, `B1_NO_FOV`, `B1_NO_POST`.
- Se verifican las identidades del freeze, selección, YOLO y los hashes de los
  cinco manifests de ablación.

El resumen usa directamente `segmenter_selection.json`: ranking, score /100,
componentes Region/35, Boundary/20, Stability/15, Reliability/15 y P0
robustness/15, además de medias B1_FULL, IQR, tasas de degeneradas/fallos y
checkpoint con SHA-256. No se aplican fórmulas nuevas.

## Revisión por caso

Por defecto se muestran Ground Truth, B1_FULL, B1_NO_FOV y B1_NO_POST. Los
controles activan B1_NO_HAIR y B1_NO_YOLO. La tabla contiene para las cinco
condiciones Dice, Jaccard, Boundary-F1 y HD95 normalizado; cada delta es
`variante - FULL`. Un delta positivo mejora Dice/Jaccard/Boundary-F1. En HD95
normalizado, menor es mejor y se muestra el valor bruto y el delta.

Se puede ordenar por |ΔDice| de NO_FOV/NO_POST, |ΔJaccard|, |ΔBoundary-F1|,
|ΔHD95|, `image_id` o `method_id`, y filtrar por método, imagen, revisado,
pendiente, existencia de observación y delta mínimo. `Siguiente pendiente`
solo prioriza trabajo; nunca excluye una imagen.

El overlay es reproducible y solo visual: combina la imagen, contorno del GT y
contorno de la condición elegida. No escribe ni altera ninguna máscara.

## Auditoría humana

Cada evaluación es comparativa por `method_id + image_id` y se registra como
una línea append-only schema 2 en
`results/benchmark_v1/review/post05_observations.jsonl`. Incluye los cinco run
IDs, hashes de manifests/imagen/máscara y las identidades congeladas, además de
severidad FOV, daño de postprocesamiento, artefactos NO_POST, problema ROI/YOLO,
preferencia visual, estado general y nota. Las decisiones son interpretativas:
no existe una opción de exclusión científica retrospectiva. Si se revisa otra
vez un caso, se añade otra línea y el endpoint conserva historial y última
observación.

La vista muestra `Revisados / 192` y `Pendientes`. Si el freeze, una identidad,
un hash o la cobertura de 192×5 no coincide, muestra un error y no permite
guardar observaciones.

La web combina runs independientes por condición y compara, por `image_id`,
`B1_FULL`, `B1_NO_FOV` y `B1_NO_POST` (también acepta `B1_NO_HAIR` y
`B1_NO_YOLO` cuando estén disponibles). La vista está limitada a `S01`, `S10`
y `S14`, conserva los previews y muestra Dice, Jaccard, Boundary-F1, HD95
normalizado y los dos deltas solicitados.

Las observaciones se guardan en
`results/benchmark_v1/review/post05_observations.jsonl`. Ese archivo es nuevo,
append-only y conserva `run_id`, `method_id`, `image_id`, condición y hashes;
no modifica `result.json`, manifests ni el scientific freeze. El deduplicado
MSKCC no se ejecuta desde esta interfaz.

## Uso local

```bash
cd "/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick_hpc_repair"
conda run --no-capture-output -n tesis-sam python scripts/serve_segmentation_review.py --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000`, seleccione los runs post-Fase 0.5 y cargue la
comparación. Ordene por delta, navegue los casos y guarde cada observación con
identidad y nota.

No se eligen runs manualmente en la ruta normal: el servidor los obtiene de la
selección y del freeze y rechaza cualquier inconsistencia. No se lanzan jobs
CEDIA para esta revisión. Si en el futuro se sincronizan artefactos nuevos,
use únicamente la ruta de resultados y vuelva a validar hashes antes de abrir
la UI:

```bash
rsync -av --exclude='*.err' --exclude='*.out' \
  <usuario>@<cedia-host>:$HOME/Thesis_Fitzpatrick/results/benchmark_v1/ \
  results/benchmark_v1/
conda run --no-capture-output -n tesis-sam python -m unittest discover -s tests -v
```

Para cualquier job Slurm futuro, excluya explícitamente `compute-0-1`:

```bash
#SBATCH --exclude=compute-0-1
```

## Pruebas locales

```bash
conda run --no-capture-output -n tesis-sam python -m unittest discover -s tests -v
```

## Commit y push manuales

```bash
cd "/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick_hpc_repair"
git status --short
git diff --check
git add scripts/serve_segmentation_review.py web/index.html web/app.js web/styles.css tests/test_review_server.py docs/POST05_VISUAL_REVIEW.md
git commit -m "Add post-Phase-0.5 visual review"
git push origin fix/hair-component-scan-performance
```

## Actualizar y probar en CEDIA por SSH

```bash
ssh <usuario>@<cedia-host>
cd "$HOME/Thesis_Fitzpatrick"
git switch fix/hair-component-scan-performance
git pull --ff-only origin fix/hair-component-scan-performance
conda run --no-capture-output -n tesis-sam python -m unittest discover -s tests -v
conda run --no-capture-output -n tesis-sam python scripts/serve_segmentation_review.py --host 0.0.0.0 --port 8000
```

Si se lanza un job CEDIA adicional, añada siempre esta directiva al script
Slurm o al comando equivalente y verifique el nodo asignado:

```bash
#SBATCH --exclude=compute-0-1
```

No ejecute el deduplicado MSKCC hasta cerrar la inspección visual y conservar
el archivo de observaciones junto con la decisión de selección.
