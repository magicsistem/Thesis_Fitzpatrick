# FASE 0: ESTADO INICIAL — Thesis_Fitzpatrick

**Fecha de corte:** 3 de agosto de 2026
**Usuario:** Miguel Benavides
**Repositorio:** <https://github.com/magicsistem/Thesis_Fitzpatrick>
**Sistema local:** CachyOS Linux, shell `fish`, Miniforge
**Versión documental:** `v01`
**Fase documentada:** `FASE 0: ESTADO INICIAL`
**Propósito de este archivo:** documentar exclusivamente el trabajo realizado hasta cerrar la Fase 0, de modo que una fase posterior pueda recibir un estado reproducible sin convertir este documento en un historial acumulativo de toda la tesis.

> Este documento distingue expresamente entre: **confirmado por salida de terminal**, **implementado/preparado**, **investigado pero no integrado** y **pendiente de comprobación**. No convertir una instrucción previa en un hecho ejecutado si no aparece como confirmada.

---

## 1. Resumen ejecutivo del estado actual

La tesis construye un flujo controlado para evaluar segmentación de lesiones dermatoscópicas y generar una máscara complementaria de piel limpia, relacionando después los resultados con el tipo de piel Fitzpatrick. El piloto actual contiene **93 imágenes**. La interfaz web permite escoger imágenes y modelos, ejecutar inferencia, recorrer los resultados como diapositivas y revisar:

- imagen original;
- máscara 1: lesión;
- máscara 2: piel limpia;
- tipo Fitzpatrick y diagnóstico disponible;
- tiempo de inferencia;
- uso de CPU;
- uso de RAM;
- resumen promedio por modelo.

La rama de trabajo es:

```text
agent/segmentation-review-ui
```

La suite llegó a **15 variantes ejecutables** con checkpoint. Se eliminó el límite artificial de diez modelos y se añadieron los controles **“Seleccionar todos”** y **“Quitar selección”**. La batería final contiene **38 pruebas**.

El error de VM-UNet causado por las claves `total_ops` y `total_params` fue corregido. Después del arreglo, **VM-UNet ISIC 2018 terminó correctamente en CPU en 9.40 s** para una imagen. Los avisos de `timm.models.layers` son advertencias de compatibilidad futura, no fallos de inferencia.

Se ejecutó la comparación de las 15 variantes sobre una muestra equilibrada de **60 imágenes: 10 por cada tipo Fitzpatrick I–VI**, equivalente a **900 pares modelo-imagen planificados**. Miguel confirmó que las máscaras se generaron para los 15 modelos y que se observaron fallos cualitativos, especialmente captura del aro o campo de visión del dermatoscopio.

No se aplicó ninguna corrección FOV a `lesion_mask.png`. La máscara derivada de piel limpia sí excluye píxeles casi negros mediante `maximum > 8`, además de reflejos, margen de lesión y pelo probable; esta regla evita usar fondo negro para estadísticas cromáticas, pero **no corrige ni recorta la predicción de lesión**.

### Estado que no debe sobreinterpretarse

- El lote comparativo confirmado cubre 60 de las 93 imágenes del piloto, no las 93.
- La auditoría local encontró **904 directorios modelo-imagen**, porque junto a los 900 pares planificados existen cuatro pruebas adicionales o parciales. No tratar 904 como el tamaño del protocolo.
- La auditoría encontró **3.375 archivos**: 1.688 PNG y 1.687 JSON. Esta cuenta documenta el estado local, pero no sustituye una validación de completitud por par.
- Los datos, checkpoints y resultados están ignorados por Git; el hito conserva código, configuración, pruebas, hashes e inventario, no duplica los artefactos pesados.
- Aún no se ha seleccionado formalmente “el mejor modelo”. La web es la herramienta para producir esa comparación.
- Las máscaras automáticas son predicciones para revisión; no deben convertirse en ground truth sin control de calidad.

---

## 2. Objetivo científico y alcance

### 2.1 Objetivo general

Construir un flujo reproducible que use metadatos de ISIC Archive para estudiar segmentación y análisis posterior de lesiones cutáneas a través de los seis tipos Fitzpatrick, con especial atención a posibles fallos en pieles de tonalidad más oscura.

### 2.2 Objetivo de la etapa actual

La etapa actual no busca todavía clasificar malignidad. Busca:

1. ejecutar diferentes segmentadores con checkpoints públicos;
2. producir una máscara binaria de lesión por modelo e imagen;
3. derivar una máscara de piel limpia;
4. registrar rendimiento computacional;
5. permitir revisión visual sistemática;
6. identificar modelos que segmenten de forma menos agresiva y sean menos sensibles a artefactos del dermatoscopio;
7. escoger posteriormente el mejor balance entre calidad y coste.

### 2.3 Distinción clínica/técnica que debe conservarse

Una lesión benigna también puede y normalmente debe ser segmentada. El segmentador responde a **“dónde está la lesión”**, no a **“es maligna”**. Por tanto:

- marcar una lesión benigna no es, por sí solo, un error de segmentación;
- confundir el aro negro, la punta transparente del dermatoscopio, reflejos o fondo con lesión sí es un error;
- malignidad/benignidad pertenece a una tarea de clasificación posterior;
- las etiquetas diagnósticas se muestran como contexto para la revisión, no como salida del segmentador.

### 2.4 FOV fuera del alcance de la Fase 0

Durante la búsqueda bibliográfica se concluyó que los bordes negros y el viñeteado suelen tratarse mediante una máscara independiente del campo de visión. La Fase 0 conserva deliberadamente las predicciones de lesión crudas para hacer visibles esos fallos. Una fase posterior podrá evaluar una solución FOV de forma separada y comparar antes/después; no atribuir a los modelos de esta fase una corrección que no recibieron.

---

## 3. Entorno local confirmado

### 3.1 Rutas y shell

```text
Sistema: CachyOS Linux
Shell: fish
HOME: /home/miguel
Miniforge: /home/miguel/miniforge3
Repositorio local: /run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick
```

El uso de comillas simples alrededor de la ruta del repositorio es obligatorio porque contiene espacios y puntos:

```fish
cd '/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick'
```

### 3.2 Entorno `tesis-sam`

Este es el entorno que aparece activo en el prompt durante la operación general del repositorio y del servidor.

Verificación ejecutada:

```fish
conda activate tesis-sam

echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
which python
python --version
which pip
pip --version
```

Salida confirmada:

```text
CONDA_DEFAULT_ENV=tesis-sam
/home/miguel/miniforge3/envs/tesis-sam/bin/python
Python 3.10.20
/home/miguel/miniforge3/envs/tesis-sam/bin/pip
pip 26.1.2 from /home/miguel/miniforge3/envs/tesis-sam/lib/python3.10/site-packages/pip (python 3.10)
```

Instalación CPU que se ejecutó en este entorno:

```fish
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
    git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2 \
    timm \
    opencv-python-headless \
    pandas
```

Versiones confirmadas después de esa instalación:

```text
torch 2.13.0+cpu
torchvision 0.28.0+cpu
torchaudio 2.11.0+cpu
numpy 2.2.6
timm 1.0.28
opencv-python-headless 5.0.0.93
pandas 2.3.3
mobile_sam 1.0
```

MobileSAM quedó fijado al commit:

```text
f706ad9c4eb7f219c00d9050e46328518ffb65d2
```

### 3.3 Entorno `thesis-avit`

Los adaptadores con dependencias más específicas se ejecutan sin activar manualmente el entorno, mediante:

```fish
conda run --no-capture-output -n thesis-avit \
    python ...
```

Hechos conocidos:

- Python 3.10.
- Ruta observada: `/home/miguel/miniforge3/envs/thesis-avit/lib/python3.10/`.
- PyTorch CPU; la configuración anterior registró PyTorch 2.12.1, `timm 1.0.28` y `einops 0.8.2`.
- Este entorno ejecuta AViT, BA-Transformer, U-Net/ResNet34, SkinMamba, Unixio, Theodore, VM-UNet y De-LightSAM.

Dependencias añadidas durante la expansión:

```fish
mamba install -n thesis-avit -c conda-forge transformers

conda run --no-capture-output -n thesis-avit \
    python -m pip install --no-deps segmentation-models-pytorch==0.5.0
```

`--no-deps` fue intencional: evita que `pip` sustituya las versiones ya verificadas de PyTorch, Torchvision y `timm`.

Comprobación de `segmentation-models-pytorch`:

```fish
conda run --no-capture-output -n thesis-avit \
    python -c 'import segmentation_models_pytorch as smp; print("SMP:", smp.__version__)'
```

---

## 4. Repositorio, ramas y Git

### 4.1 Origen

```text
GitHub: https://github.com/magicsistem/Thesis_Fitzpatrick
```

El repositorio fue inicializado/clonado en la ruta local indicada. El primer estado registrado tuvo la rama `main` y el commit inicial:

```text
eae3341
```

Ese commit fue enviado a `origin/main`.

### 4.2 Rama actual

```text
agent/segmentation-review-ui
```

Estado confirmado antes del commit documental de cierre:

```text
a6e0abb (HEAD -> agent/segmentation-review-ui, origin/agent/segmentation-review-ui) Allow all-model review and fix VM-UNet loading
5698c40 Add expanded verified segmentation model suite
0d796f9 Document dermoscopy segmentation model search
8ef97cc Add verified SkinMamba suite and checkpoint audit
55ba32c Handle empty skin masks and add U-Net baseline
b606d35 Make BA-Transformer loading portable
55eddb3 Add stratified review and BA-Transformer adapter
f611bc0 Fix CPU reporting and add UltraLight adapter
60733b5 Add reproducible AViT CPU adapter
e4de2c8 Refresh model catalog and benchmark resources
76ace15 Add segmentation mask review workflow
a5b0829 (main, origin/main) Add MobileSAM prompt pilot and findings documentation
```

GitHub confirmó que la rama estaba **11 commits por delante de `main` y 0 por detrás**. La rama se publicó correctamente con seguimiento remoto mediante:

```fish
git push -u origin agent/segmentation-review-ui
```

El commit `a6e0abb` es la base funcional auditada. El commit que incorpora este handoff y la limpieza documental será posterior; el hito canónico del cierre se identifica mediante la etiqueta anotada `fase-0-estado-inicial` sobre el resultado integrado en `main`.

### 4.3 Comandos para reconstruir el estado Git real al abrir un chat nuevo

```fish
cd '/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick'

git status --short --branch
git log -10 --oneline --decorate
git remote -v
```

Si se necesita comprobar archivos modificados:

```fish
git diff --stat
git diff --check
```

La rama ya está publicada. Para comprobar que el checkout local conserva el seguimiento remoto:

```fish
git branch -vv
```

### 4.4 Acuerdo de trabajo

Para cambios simples, no duplicar innecesariamente la misma revisión antes y después del commit. Se acordó revisar una vez con suficiente profundidad: o antes del commit o desde GitHub después del push, según el flujo concreto.

---

## 5. Datos y muestra piloto

### 5.1 Conjunto actual

El piloto contiene **93 imágenes** y se usa para una evaluación controlada estratificada por tipos Fitzpatrick I–VI.

Directorio de inferencia:

```text
data/processed/isic_segmentation_pilot_resized_inference/
```

Imagen usada para la prueba de modelos pesados:

```text
data/processed/isic_segmentation_pilot_resized_inference/ISIC_0477738.jpg
```

Otra imagen usada previamente para una prueba individual de U-Net/ResNet34:

```text
data/processed/isic_segmentation_pilot_resized_inference/ISIC_1136232.jpg
```

### 5.2 Estratificación

La lógica de muestra estratificada equilibra los seis tipos Fitzpatrick. Una prueba automatizada exige que el tamaño solicitado sea múltiplo de seis para mantener el balance exacto.

### 5.3 Metadatos mostrados

La revisión incluye cuando están disponibles:

- identificador/nombre de imagen;
- tipo Fitzpatrick;
- diagnóstico;
- otros metadatos que ya estén presentes en el estado del servidor.

No inferir el diagnóstico a partir de la máscara.

### 5.4 Lote comparativo ejecutado e inventario local

Protocolo informado por Miguel:

```text
15 variantes × 60 imágenes equilibradas = 900 pares modelo-imagen
60 imágenes = 10 por cada tipo Fitzpatrick I, II, III, IV, V y VI
```

Auditoría del árbol local de resultados:

```text
Tamaño de data/:                 189 MB
Tamaño de models/:              1,6 GB
Tamaño de reports/tables/:      228 KB
Tamaño de results/:              45 MB
Variantes con resultados:           15
Directorios modelo-imagen:          904
Archivos del benchmark:           3.375
PNG:                              1.688
JSON:                             1.687
```

Los cuatro directorios adicionales sobre los 900 planificados corresponden a pruebas individuales o parciales efectuadas durante la integración. Los archivos locales permanecen deliberadamente ignorados por Git. No volver a ejecutar el lote solo para respaldarlo en el repositorio: conservar el inventario y regenerar únicamente cuando el protocolo de una fase posterior lo requiera.

---

## 6. Comportamiento de la interfaz web

### 6.1 Flujo de usuario

1. Iniciar el servidor local.
2. Abrir la web en `http://127.0.0.1:8000`.
3. Seleccionar uno, varios o todos los modelos.
4. Seleccionar desde una imagen hasta el máximo disponible.
5. Ejecutar el lote.
6. Navegar en orden por modelo y por imagen.
7. Revisar original, lesión, piel limpia, metadatos y rendimiento.
8. Comparar el resumen promedio de cada modelo.

### 6.2 Inicio del servidor

Comando canónico:

```fish
cd '/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick'

conda activate tesis-sam

python scripts/serve_segmentation_review.py \
    --host 127.0.0.1 \
    --port 8000
```

Forma equivalente en una sola línea:

```fish
python scripts/serve_segmentation_review.py --host 127.0.0.1 --port 8000
```

Si el puerto 8000 está ocupado por una instancia anterior, ir a esa terminal, pulsar `Ctrl+C` y ejecutar de nuevo el comando.

### 6.3 Salidas por imagen

- **Máscara 1:** máscara binaria de lesión entregada por el adaptador.
- **Máscara 2:** piel limpia derivada excluyendo lesión y artefactos contemplados por la lógica actual.
- **Rendimiento:** tiempo, CPU y RAM.
- **Estadísticas de piel limpia:** RGB, Lab e ITA cuando quedan píxeles válidos.

### 6.4 Corrección de máscara de piel limpia vacía

Error original:

```text
The clean-skin mask contains no pixels.
```

Causa: una predicción podía cubrir toda la imagen y dejar la máscara complementaria sin píxeles.

Comportamiento corregido:

- el resultado permanece en estado `ready`;
- se conservan y muestran ambas máscaras;
- se conservan tiempo/CPU/RAM;
- solo RGB/Lab/ITA se marcan como no disponibles;
- un modelo con esa predicción no bloquea el resto del lote.

Prueba de regresión relevante:

```text
test_full_lesion_mask_keeps_result_ready_without_skin_statistics
```

### 6.5 Selección sin límite de diez

Estado final:

- el servidor acepta los 15 IDs en una sola petición;
- se eliminó la validación fija de diez;
- la interfaz muestra `Seleccionar todos` y `Quitar selección`;
- una prueba HTTP automatizada confirmó que el servidor devuelve 15 resultados sin rechazo.

La ausencia de límite no significa que el lote sea rápido. VM-UNet, SkinMamba y De-LightSAM pueden aumentar mucho el tiempo total, sobre todo con muchas imágenes.

### 6.6 Reutilización de resultados

Las máscaras ya calculadas se reutilizan cuando corresponde. Reiniciar el servidor no implica necesariamente recalcular todos los resultados existentes.

---

## 7. Arquitectura del código relevante

Los nombres siguientes aparecen en los cambios realizados y constituyen el mapa mínimo para continuar:

```text
configs/segmentation_models.json
configs/isic_fitzpatrick_study.toml
scripts/serve_segmentation_review.py
scripts/adapters/
scripts/setup_unet_resnet34.py
scripts/setup_skinmamba.py
scripts/setup_unixio_isic2018.py
scripts/setup_theodore_isic2018.py
scripts/setup_vmunet.py
scripts/setup_delightsam.py
src/thesis_fitzpatrick/
tests/
web/app.js
README.md
docs/SEGMENTATION_REVIEW_UI.md
docs/SEGMENTATION_CHECKPOINT_AUDIT.md
docs/DERMOSCOPY_SEGMENTATION_MODEL_SEARCH_2026.md
```

Responsabilidades:

- `configs/segmentation_models.json`: catálogo ejecutable, comando/adaptador, checkpoint, clase de coste, licencia y metadatos visibles.
- `scripts/serve_segmentation_review.py`: API local, ejecución de adaptadores, caché/resultados, métricas y validación de selección.
- `scripts/adapters/*.py`: adaptación de cada repositorio/checkpoint a una interfaz uniforme de imagen de entrada y PNG binario de salida.
- `scripts/setup_*.py`: descarga reproducible de fuente y checkpoint, fijación de revisión y comprobación de hash.
- `web/app.js`: selector, envío de modelos/imágenes, navegación y controles de seleccionar/quitar todos.
- `tests/`: contratos de preprocesamiento, hashes, carga de checkpoints, rutas, servidor, métricas y regresiones.

---

## 8. Suite ejecutable actual: 15 variantes

| # | Variante visible | Familia/origen | Dataset del checkpoint | Estado | Coste aproximado | Licencia conocida |
|---:|---|---|---|---|---|---|
| 1 | AViT | ViT para lesión cutánea | ISIC | Ejecutado en el lote de Fase 0 | Pesado | No declarada |
| 2 | UltraLight VM-UNet | Mamba ultraligero | Dermatoscopia | Ejecutado en el lote de Fase 0 | Ligero | MIT |
| 3 | BA-Transformer | Boundary-aware transformer | ISIC 2016 | Ejecutado en el lote de Fase 0 | Pesado | No declarada |
| 4 | U-Net/ResNet34 | `segmentation-models-pytorch` | ISIC 2018 | Ejecutado en el lote de Fase 0 | Normal | MIT |
| 5 | SkinMamba ISIC 2017 | CNN/Mamba | ISIC 2017 | Ejecutado en el lote de Fase 0 | Pesado | Apache-2.0 |
| 6 | SkinMamba ISIC 2018 | CNN/Mamba | ISIC 2018 | Ejecutado en el lote de Fase 0 | Pesado | Apache-2.0 |
| 7 | Unixio U-Net | U-Net | ISIC 2018 | Ejecutado en el lote de Fase 0 | Normal | MIT |
| 8 | Unixio U-Net++ | U-Net++ | ISIC 2018 | Ejecutado en el lote de Fase 0 | Normal/pesado | MIT |
| 9 | Unixio Attention U-Net | Attention U-Net | ISIC 2018 | Ejecutado en el lote de Fase 0 | Pesado | MIT |
| 10 | Theodore U-Net | U-Net | ISIC 2018 | Ejecutado en el lote de Fase 0 | Normal | MIT |
| 11 | Theodore Inception | Inception segmentation | ISIC 2018 | Ejecutado en el lote de Fase 0 | Normal/pesado | MIT |
| 12 | Theodore SegFormer-B0 | Transformer | ISIC 2018 | Ejecutado en el lote de Fase 0 | Normal/pesado | MIT |
| 13 | VM-UNet ISIC 2017 | Vision Mamba U-Net | ISIC 2017 | Corregido y ejecutado en el lote de Fase 0 | Muy pesado | Apache-2.0 |
| 14 | VM-UNet ISIC 2018 | Vision Mamba U-Net | ISIC 2018 | Corregido y ejecutado en el lote de Fase 0 | Muy pesado | Apache-2.0 |
| 15 | De-LightSAM Dermoscopy | SAM ligero adaptado | Dermoscopy/ISIC | Ejecutado en el lote de Fase 0 | Muy pesado por entrada 1024 | Apache-2.0 |

### 8.1 Checkpoints y rutas confirmadas

#### AViT

```text
models/avit/checkpoints/AViT_ViT_B_ISIC_best.pth
```

#### BA-Transformer

```text
models/ba-transformer/checkpoints/best_isic2016.pkl
```

#### U-Net/ResNet34

```text
models/unet-resnet34/checkpoints/best_unet_resnet34.pth
```

SHA-256 esperado:

```text
1ea87e341552768234b367c3b68704030bc1bc08991c323508ea5c5086d9d334
```

#### SkinMamba

```text
models/skinmamba/checkpoints/isic17.pth
models/skinmamba/checkpoints/isic18.pth
```

Hashes esperados:

```text
ISIC 2017: 9b941f1459dd6e7660bd5ec32b2b58ba861c596ac8b14708a43a9c940f0645a7
ISIC 2018: 36a2c352cd39011db3f416373be1bdd98b079d031fc778dbc640780823388543
```

Commit fijado del código SkinMamba:

```text
131a85da14da4d3bf135b52912d274191c05f3b8
```

#### Unixio Attention U-Net

```text
models/unixio-isic2018/checkpoints/best_attention_unet.pt
```

#### VM-UNet

```text
models/vmunet/checkpoints/best-vmunet-isic17.pth
models/vmunet/checkpoints/best-vmunet-isic18.pth
```

#### De-LightSAM

```text
models/delightsam/checkpoints/ESP_isic_best.pth
```

Para las demás rutas exactas no copiadas en este historial, usar como fuente de verdad:

```fish
python -m json.tool configs/segmentation_models.json
```

No adivinar nombres de checkpoints si el catálogo ya los declara.

### 8.2 Verificaciones técnicas realizadas al implementar

- checkpoint descargable y tamaño razonable;
- SHA-256 fijado cuando fue posible;
- estructura interna compatible con la arquitectura reconstruida;
- normalización y resolución de entrada contrastadas con el repositorio original;
- carga estricta `strict=True` cuando la arquitectura lo permite;
- salida convertida a PNG binario con valores `[0, 255]`;
- ejecución CPU o adaptación de llamadas que forzaban CUDA;
- pruebas unitarias de hashes, preprocesamiento y limpieza del `state_dict`;
- metadatos de licencia y procedencia visibles, incluyendo “no declarada” cuando corresponde.

### 8.3 SkinMamba: validación adicional

Se confirmó:

- dos checkpoints oficiales reales, ISIC 2017 e ISIC 2018;
- carga con `strict=True`;
- **14,083,201 parámetros**;
- sustitución/implementación CPU de la recurrencia selective-scan;
- coincidencia numérica con la referencia con error máximo aproximado de `1.9e-6`;
- inferencia completa que produjo una máscara PNG binaria válida.

### 8.4 De-LightSAM: adaptación CPU

El ZIP oficial contenía un checkpoint dermatoscópico de aproximadamente 49 MB con hash estable. Se eliminó únicamente la suposición de CUDA en el script de evaluación, manteniendo el modelo y la resolución de entrada de 1024×1024. No se añadió FOV.

### 8.5 Validación aislada de la expansión

Se confirmó carga estricta e inferencia binaria real para:

- Theodore U-Net;
- Theodore Inception;
- Unixio Attention U-Net.

SegFormer, U-Net++/ResNet34, VM-UNet y De-LightSAM quedaron inicialmente cubiertos por estructura, hashes y pruebas unitarias. Después, VM-UNet ISIC 2018 y De-LightSAM también se ejecutaron en el equipo de Miguel.

---

## 9. Preparación reproducible de modelos

Desde la raíz del repositorio:

```fish
cd '/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick'
```

### 9.1 U-Net/ResNet34

```fish
conda run --no-capture-output -n thesis-avit \
    python -m pip install --no-deps \
    segmentation-models-pytorch==0.5.0

conda run --no-capture-output -n thesis-avit \
    python scripts/setup_unet_resnet34.py

sha256sum models/unet-resnet34/checkpoints/best_unet_resnet34.pth
du -h models/unet-resnet34/checkpoints/best_unet_resnet34.pth
```

### 9.2 SkinMamba

```fish
conda run --no-capture-output -n thesis-avit \
    python scripts/setup_skinmamba.py
```

Verificación:

```fish
sha256sum \
    models/unet-resnet34/checkpoints/best_unet_resnet34.pth \
    models/skinmamba/checkpoints/isic17.pth \
    models/skinmamba/checkpoints/isic18.pth

git -C models/skinmamba/source rev-parse HEAD
```

### 9.3 Suite ampliada

```fish
mamba install -n thesis-avit -c conda-forge transformers

conda run --no-capture-output -n thesis-avit \
    python -m pip install --no-deps segmentation-models-pytorch==0.5.0

conda run --no-capture-output -n thesis-avit \
    python scripts/setup_unixio_isic2018.py

conda run --no-capture-output -n thesis-avit \
    python scripts/setup_theodore_isic2018.py

conda run --no-capture-output -n thesis-avit \
    python scripts/setup_vmunet.py

conda run --no-capture-output -n thesis-avit \
    python scripts/setup_delightsam.py
```

Estos comandos ya fueron entregados y los modelos correspondientes aparecieron después en pruebas reales. Si un chat nuevo duda si todos los instaladores terminaron, no reinstalar de inmediato: primero comprobar archivos y hashes.

---

## 10. Modelos investigados pero no integrados en la suite actual

La regla final fue estricta: un modelo solo aparece en la web si tiene checkpoint recuperable, arquitectura reconstruible y ruta de inferencia reproducible. La licencia puede estar sin declarar, porque Miguel autorizó probar esos modelos, pero la ausencia de licencia debe seguir visible.

| Modelo/familia | Resultado de la investigación | Decisión actual |
|---|---|---|
| ISCF | El repositorio anunciaba pesos ISIC 2017/2018, pero los enlaces oficiales de Drive dejaron de permitir recuperar los checkpoints de manera reproducible. | Fuera del selector; conservar en auditoría histórica. |
| DermoSegDiff-A | Método MIT y consciente de fronteras, prometedor para bordes ambiguos. El enlace oficial de SharePoint respondió `401`. | No integrado mientras no exista descarga reproducible. |
| DevBhuyan “DenseNet201 U-Net” | El archivo H5 descargado fue inspeccionado y resultó ser un clasificador ResNet50, no un segmentador. | Excluido; no usar como segmentación. |
| EGE-UNet | Código oficial Apache-2.0, eficiente, pero sin checkpoint público verificable en la auditoría. | No integrado; candidato de reentrenamiento. |
| UCM-Net | Código oficial Apache-2.0, muy ligero, pero sin checkpoint público verificable. | No integrado; candidato de reentrenamiento. |
| MALUNet | Arquitectura ligera prometedora; no se verificó un checkpoint utilizable. | No integrado. |
| MHorUNet | Código MIT; disponibilidad de checkpoint no quedó demostrada de forma reproducible. | No integrado. |
| HSH-UNet | Código MIT, sin checkpoint recuperable confirmado. | No integrado. |
| EM-Net | El repositorio incluye pesos de demostración que sus autores dicen no usar directamente; pesos específicos se solicitan por correo. | No automatizable; no integrado. |
| SET | Código/paper interesante con priors de superpíxeles; no se obtuvo checkpoint ejecutable. | No integrado. |
| MDViT | Arquitectura multidominio, sin pesos públicos confirmados en la revisión. | No integrado. |
| TMUNet | La investigación inicial reportó enlaces a pesos para ISIC17/18/PH2, pero no llegó a la fase de verificación local/hash dentro de esta suite. | No aparece entre los 15; requiere auditoría técnica antes de integrarlo. |
| Attention DeepLabv3+ | Repositorio clásico con pesos enlazados; no fue llevado al adaptador actual. | Pendiente de verificación técnica, no implementado. |
| BCDU-Net | Baseline Keras/TensorFlow con pesos anunciados; no fue integrado al entorno PyTorch actual. | No implementado. |
| DSNet | Código disponible, sin checkpoint público confirmado. | No integrado. |
| Superpixel Merging | Baseline clásico CPU, sin checkpoint porque no es una red preentrenada. | Fuera de la suite de checkpoints. |
| BiomedParse | Fue considerado durante la búsqueda amplia; el coste/flujo y la falta de una integración dermatoscópica verificada impidieron incorporarlo. | No integrado. |

La auditoría histórica se documentó en:

```text
docs/SEGMENTATION_CHECKPOINT_AUDIT.md
docs/DERMOSCOPY_SEGMENTATION_MODEL_SEARCH_2026.md
```

El informe inicial de modelos también priorizó TMUNet, AViT y EGE-UNet/UCM-Net para diferentes usos. Esa recomendación bibliográfica no equivale a que estén todos implementados.

---

## 11. Historial de parches, hashes y aplicación

### 11.1 U-Net y máscara vacía

Parche original:

```text
0001-Handle-empty-skin-masks-and-add-U-Net-baseline.patch
SHA-256: 7f00779842c7138cb9b866661ac0c6f7270a8019e574278c571d3d97b3d9609a
```

Este cambio quedó reflejado en:

```text
55ba32c Handle empty skin masks and add U-Net baseline
```

### 11.2 Parche acumulativo que falló

Archivo:

```text
0001-Add-verified-segmentation-checkpoint-suite.patch
SHA-256: 25330bba3228365e1d1d7192f09c9b770f5789412df912ea4f932699ac8db781
```

Comando ejecutado:

```fish
sha256sum ~/Downloads/0001-Add-verified-segmentation-checkpoint-suite.patch
git am ~/Downloads/0001-Add-verified-segmentation-checkpoint-suite.patch
```

Falló porque intentaba volver a añadir U-Net y la corrección de máscara vacía, que ya estaban presentes. Los conflictos incluyeron:

```text
README.md
configs/isic_fitzpatrick_study.toml
configs/segmentation_models.json
docs/SEGMENTATION_REVIEW_UI.md
scripts/adapters/unet_resnet34.py
scripts/serve_segmentation_review.py
scripts/setup_unet_resnet34.py
tests/test_review_server.py
tests/test_unet_resnet34_adapter.py
web/app.js
```

No se resolvió manualmente. Se abortó correctamente:

```fish
git am --abort
git status --short --branch
```

Salida confirmada:

```text
## agent/segmentation-review-ui
```

### 11.3 Parche incremental SkinMamba

```text
0001-Add-verified-SkinMamba-suite-and-checkpoint-audit.patch
SHA-256: 73a0f1574ea6ff38da5dc892c972bf4c36868118d3a82c84faf6aa0f25a0c546
```

Aplicación ejecutada:

```fish
sha256sum ~/Downloads/0001-Add-verified-SkinMamba-suite-and-checkpoint-audit.patch

git am ~/Downloads/0001-Add-verified-SkinMamba-suite-and-checkpoint-audit.patch
```

Resultado:

```text
Applying: Add verified SkinMamba suite and checkpoint audit
```

Commit:

```text
8ef97cc Add verified SkinMamba suite and checkpoint audit
```

### 11.4 Informe de búsqueda amplia

```text
0001-Document-dermoscopy-segmentation-model-search.patch
SHA-256: 13f112c69dc96b3fb58b50132d966d80116d9544ed49d62962473d00b46c11db
```

Comandos entregados:

```fish
sha256sum ~/Downloads/0001-Document-dermoscopy-segmentation-model-search.patch

git am ~/Downloads/0001-Document-dermoscopy-segmentation-model-search.patch

git status --short --branch
git log -1 --oneline
```

### 11.5 Expansión a 15 variantes

```text
0001-Add-expanded-verified-segmentation-model-suite.patch
SHA-256: 15f51dd2439df71399a724605e26ce6870516506e7b126909d2b2d9e76f5b4d5
```

Aplicación indicada:

```fish
git status --short --branch

sha256sum ~/Downloads/0001-Add-expanded-verified-segmentation-model-suite.patch

git am ~/Downloads/0001-Add-expanded-verified-segmentation-model-suite.patch
```

La aplicación no aparece copiada literalmente, pero su contenido está funcionalmente confirmado porque después se ejecutaron adaptadores añadidos por ese parche: Unixio Attention U-Net, VM-UNet y De-LightSAM.

### 11.6 Selección de todos y corrección VM-UNet

```text
0001-Allow-all-model-review-and-fix-VM-UNet-loading.patch
SHA-256: e8d8a8d0ab7fa4384dcc100f7fc1ece95ceed2043632c9d0af9642280fc7324d
```

Aplicación indicada:

```fish
sha256sum ~/Downloads/0001-Allow-all-model-review-and-fix-VM-UNet-loading.patch

git am ~/Downloads/0001-Allow-all-model-review-and-fix-VM-UNet-loading.patch

python -m unittest discover -s tests -v
python -m py_compile scripts/*.py scripts/adapters/*.py src/thesis_fitzpatrick/*.py

git status --short --branch
git log -3 --oneline
```

Su funcionamiento quedó confirmado porque VM-UNet ISIC 2018 terminó después de aplicar la corrección.

---

## 12. Pruebas automatizadas y validaciones

### 12.1 Comandos canónicos

```fish
python -m unittest discover -s tests -v

python -m py_compile \
    scripts/*.py \
    scripts/adapters/*.py \
    src/thesis_fitzpatrick/*.py

python -m json.tool configs/segmentation_models.json >/dev/null
and echo "JSON válido"

git status --short --branch
git log -3 --oneline
```

En `fish`, `and echo "JSON válido"` solo se ejecuta si la validación JSON termina correctamente.

### 12.2 Evolución del número de pruebas

```text
24 pruebas: U-Net/ResNet34 + manejo de máscara vacía.
27 pruebas: incorporación de SkinMamba y catálogo de seis variantes.
37 pruebas: expansión a 15 variantes.
38 pruebas: corrección de VM-UNet y eliminación del límite de diez.
```

### 12.3 Ejecución confirmada de 27 pruebas

Después de aplicar el parche incremental de SkinMamba se ejecutó:

```fish
python -m unittest discover -s tests -v
python -m py_compile scripts/*.py scripts/adapters/*.py src/thesis_fitzpatrick/*.py
python -m json.tool configs/segmentation_models.json >/dev/null
git status --short --branch
git log -3 --oneline
```

Resultado confirmado:

```text
Ran 27 tests in 0.615s

OK
```

Pruebas relevantes que pasaron:

```text
AViT: normalización ImageNet y hash publicado.
BA-Transformer: limpieza de DataParallel, bypass de ruta ResNet absoluta, BGR 0–1 y hash.
Máscaras: resumen de color y exclusión de lesión/borde negro/highlight.
Rutas: rechazo de directorios arbitrarios y raíces protegidas.
Servidor: métricas, placeholders, promedios, máscara vacía, índice, resultados y estado.
Estratificación: balance I–VI y exigencia de múltiplos de seis.
SkinMamba: hashes, limpieza de buffers de profiling y preprocesamiento.
UltraLight: normalización min-max y hash.
U-Net/ResNet34: normalización ImageNet y hash.
```

### 12.4 Suite final

El 3 de agosto de 2026 se repitió la suite en el checkout local real, con `tesis-sam` y Python 3.10.20:

```text
Ran 38 tests in 0.862s

OK
Compilación Python: OK
Catálogo JSON: OK
git diff --check: OK
```

El servidor reconoció las 15 variantes ejecutables. `git status --short --branch` no mostró cambios ni archivos sin seguimiento en ese momento. Estas comprobaciones validaron el commit funcional `a6e0abb`; el cierre documental posterior debe repetirlas antes de publicar el PR.

---

## 13. Prueba real de modelos pesados sobre una imagen

### 13.1 Preparación ejecutada

```fish
cd '/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick'

set image data/processed/isic_segmentation_pilot_resized_inference/ISIC_0477738.jpg
set output_dir /tmp/heavy-segmentation-smoke

mkdir -p $output_dir

test -f $image
and echo "Imagen: $image"
```

Salida:

```text
Imagen: data/processed/isic_segmentation_pilot_resized_inference/ISIC_0477738.jpg
```

### 13.2 BA-Transformer — confirmado

```fish
time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/ba_transformer.py \
    --image $image \
    --output $output_dir/ba-transformer.png \
    --source models/ba-transformer/source \
    --checkpoint models/ba-transformer/checkpoints/best_isic2016.pkl
```

Salida relevante:

```text
pretrained resnet, 50
Executed in 3.76 secs
```

### 13.3 AViT — confirmado

```fish
time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/avit.py \
    --image $image \
    --output $output_dir/avit.png \
    --source models/avit/source \
    --checkpoint models/avit/checkpoints/AViT_ViT_B_ISIC_best.pth \
    --device cpu
```

Resultado:

```text
Executed in 4.74 secs
```

Apareció:

```text
FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
```

No es un error.

### 13.4 Unixio Attention U-Net — confirmado

```fish
time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/unixio_isic2018.py \
    --image $image \
    --output $output_dir/unixio-attention-unet.png \
    --checkpoint models/unixio-isic2018/checkpoints/best_attention_unet.pt \
    --variant attention_unet
```

Resultado:

```text
Executed in 2.76 secs
```

### 13.5 VM-UNet ISIC 2018 — primer intento fallido

```fish
time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/vmunet.py \
    --image $image \
    --output $output_dir/vmunet-isic18.png \
    --source models/vmunet/source \
    --checkpoint models/vmunet/checkpoints/best-vmunet-isic18.pth \
    --dataset isic18
```

Error:

```text
RuntimeError: Error(s) in loading state_dict for VMUNet:
    Unexpected key(s) in state_dict: "total_ops", "total_params".
```

### 13.6 VM-UNet ISIC 2017 — primer intento fallido

```fish
time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/vmunet.py \
    --image $image \
    --output $output_dir/vmunet-isic17.png \
    --source models/vmunet/source \
    --checkpoint models/vmunet/checkpoints/best-vmunet-isic17.pth \
    --dataset isic17
```

Presentó el mismo error de `total_ops` y `total_params`.

### 13.7 Causa y corrección de VM-UNet

Los checkpoints guardaban contadores generados por herramientas de profiling:

```text
total_ops
total_params
```

El filtro existente solo eliminaba esos nombres cuando aparecían dentro de módulos. Los dos checkpoints también los tenían en la raíz del `state_dict`. La corrección elimina tanto las claves raíz como las anidadas antes de llamar:

```python
model.load_state_dict(..., strict=True)
```

Se añadió una regresión automatizada para ambos casos.

### 13.8 VM-UNet ISIC 2018 — prueba posterior al arreglo confirmada

```fish
set image data/processed/isic_segmentation_pilot_resized_inference/ISIC_0477738.jpg

time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/vmunet.py \
    --image $image \
    --output /tmp/vmunet-isic18-fixed.png \
    --source models/vmunet/source \
    --checkpoint models/vmunet/checkpoints/best-vmunet-isic18.pth \
    --dataset isic18
```

Resultado confirmado:

```text
Executed in 9.40 secs
usr time 27.86 secs
sys time 0.99 secs
```

No hubo traceback. El `FutureWarning` de `timm` es inocuo.

Validación de la máscara indicada:

```fish
python -c '
from PIL import Image
import numpy as np

path = "/tmp/vmunet-isic18-fixed.png"
mask = np.asarray(Image.open(path).convert("L"))
positive = mask >= 128

print("Tamaño:", mask.shape)
print("Valores:", np.unique(mask).tolist())
print("Píxeles de lesión:", int(positive.sum()))
print("Cobertura:", round(float(positive.mean() * 100), 4), "%")
'
```

La salida de este script puntual no fue copiada al historial. El lote posterior sí generó máscaras de VM-UNet ISIC 2018 dentro del benchmark, por lo que no es necesario repetir la inferencia únicamente para reconstruir la salida temporal de `/tmp`; los valores numéricos de ese PNG específico permanecen no documentados.

### 13.9 De-LightSAM — confirmado

```fish
time conda run --no-capture-output -n thesis-avit \
    python scripts/adapters/delightsam.py \
    --image $image \
    --output $output_dir/delightsam.png \
    --source models/delightsam/source \
    --checkpoint models/delightsam/checkpoints/ESP_isic_best.pth
```

Resultado:

```text
Executed in 4.44 secs
```

También imprimió una lista `LR SCALES`. No fue un error.

### 13.10 Inspección de máscaras ejecutada

```fish
python -c '
from pathlib import Path
from PIL import Image
import numpy as np

directory = Path("/tmp/heavy-segmentation-smoke")

for path in sorted(directory.glob("*.png")):
    mask = np.asarray(Image.open(path).convert("L"))
    positive = mask >= 128
    print(
        f"{path.name:28} "
        f"tamaño={mask.shape} "
        f"valores={np.unique(mask).tolist()} "
        f"lesión={positive.sum():8d} píxeles "
        f"cobertura={positive.mean() * 100:8.4f}%"
    )
'
```

Resultados confirmados antes de corregir VM-UNet:

```text
avit.png                     tamaño=(768, 1024) valores=[0, 255] lesión=   19865 píxeles cobertura=  2.5260%
ba-transformer.png           tamaño=(768, 1024) valores=[0, 255] lesión=  156413 píxeles cobertura= 19.8889%
delightsam.png               tamaño=(768, 1024) valores=[0, 255] lesión=   16426 píxeles cobertura=  2.0887%
unixio-attention-unet.png    tamaño=(768, 1024) valores=[0, 255] lesión=   13889 píxeles cobertura=  1.7661%
```

La gran diferencia de BA-Transformer —19.8889% frente a aproximadamente 1.77–2.53% en los otros tres— es precisamente el tipo de diferencia que debe revisarse visualmente. La cobertura sola no demuestra cuál máscara es correcta.

---

## 14. Errores resueltos y advertencias conocidas

| Problema | Causa | Solución | Estado |
|---|---|---|---|
| `The clean-skin mask contains no pixels` | Predicción cubría toda la imagen. | Mantener resultado `ready`; omitir solo estadísticas de color. | Resuelto y probado. |
| BA-Transformer dependía de una ruta absoluta del ResNet del autor | Constructor no portable. | Bypass/ajuste de la ruta absoluta y normalización del checkpoint. | Resuelto en `b606d35`. |
| Parche acumulativo no aplicaba | U-Net y máscara vacía ya existían. | `git am --abort` y parche incremental. | Resuelto. |
| VM-UNet: claves inesperadas `total_ops`, `total_params` | Buffers de profiling en raíz y módulos. | Filtrar ambos niveles antes de carga estricta. | Resuelto; ISIC17 e ISIC18 ejecutados en el lote de Fase 0. |
| `timm.models.layers` deprecated | API antigua usada por repositorios externos. | Ninguna acción necesaria para la inferencia actual. | Advertencia inocua. |
| `timm.models.registry` deprecated | API antigua en De-LightSAM. | Ninguna acción necesaria ahora. | Advertencia inocua. |
| DermoSegDiff SharePoint `401` | Recurso oficial ya no accesible. | No integrar sin fuente verificable. | Bloqueado externamente. |
| DevBhuyan H5 no era segmentador | Contenido real: clasificador ResNet50. | Excluir. | Resuelto por auditoría. |

---

## 15. Procedimiento operativo al recibir la Fase 0

### Paso 1 — Confirmar el hito, no reconstruirlo

```fish
cd '/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick'

conda activate tesis-sam

git status --short --branch
git log -12 --oneline --decorate
git branch -vv
git tag --list --format='%(refname:short) | %(objectname:short) | %(subject)'
```

Tras integrar el cierre, `main` debe contener este documento y la etiqueta
`fase-0-estado-inicial`. La rama `agent/segmentation-review-ui` puede seguir
usándose para desarrollar la fase siguiente después de sincronizarla con
`main`.

### Paso 2 — Verificar el código antes de una nueva fase

```fish
python -m unittest discover -s tests -v

python -m py_compile \
    scripts/*.py \
    scripts/adapters/*.py \
    src/thesis_fitzpatrick/*.py

python -m json.tool configs/segmentation_models.json >/dev/null
and echo "Catálogo JSON: OK"

git diff --check
and echo "git diff --check: OK"
```

Se esperan 38 pruebas y 15 variantes mientras la nueva fase no cambie esos
contratos.

### Paso 3 — Conservar los resultados de Fase 0

No volver a ejecutar automáticamente los 900 pares. Primero comprobar:

```fish
set benchmark_root results/segmentation_benchmark

find $benchmark_root -mindepth 1 -maxdepth 1 -type d | count
find $benchmark_root -mindepth 2 -maxdepth 2 -type d | count
find $benchmark_root -type f | count
```

El inventario de cierre fue 15 variantes, 904 directorios modelo-imagen y 3.375
archivos. Una fase posterior que aplique FOV debe escribir resultados en una
ruta o variante claramente distinta para conservar la comparación antes/después.

### Paso 4 — Iniciar la web únicamente cuando se necesite revisar resultados

```fish
python scripts/serve_segmentation_review.py --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000`. La interfaz reutiliza máscaras existentes y no
recalcula un par cuyo `lesion_mask.png` ya esté presente.

---

## 16. Estado entregado a la fase siguiente

### Confirmado y cerrado en Fase 0

- Rama funcional publicada en `origin/agent/segmentation-review-ui`.
- Base funcional `a6e0abb`, 11 commits por delante del `main` recibido.
- 15 variantes con checkpoint, fuente, licencia, adaptador y hash documentados.
- 38/38 pruebas aprobadas en `tesis-sam` con Python 3.10.20.
- Compilación Python, JSON y `git diff --check` aprobados en el estado funcional.
- Muestra equilibrada de 60 imágenes, 10 por tipo Fitzpatrick I–VI.
- 900 pares modelo-imagen planificados y máscaras generadas según la ejecución
  informada por Miguel.
- Inventario local: 904 directorios modelo-imagen, 3.375 archivos y 45 MB de
  resultados.
- Artefactos pesados excluidos correctamente de Git.
- Fallos cualitativos visibles, especialmente captura del aro/FOV.

### Pendientes transferidos, no realizados en Fase 0

- Definir y evaluar una corrección de campo de visión sin sobrescribir las
  predicciones crudas.
- Establecer una rúbrica visual y cuantitativa homogénea.
- Vincular las imágenes con máscaras de referencia cuando existan y calcular
  Dice, IoU, Boundary F1 y HD95.
- Cuantificar falsos positivos fuera del FOV, máscaras vacías/casi completas y
  fallos por tipo Fitzpatrick.
- Seleccionar formalmente el mejor balance de calidad, tiempo, CPU y RAM.
- Clasificación benigna/maligna, reentrenamiento, optimización y despliegue.

---

## 17. Reglas para el siguiente asistente

1. Leer este documento antes de proponer comandos.
2. Tratar `fase-0-estado-inicial` como referencia inmutable y registrar en otro `.md` solo el trabajo de la nueva fase.
3. Usar sintaxis `fish`, no Bash, para variables interactivas:

   ```fish
   set image ruta/a/imagen.jpg
   ```

4. Mantener `tesis-sam` para el servidor y `conda run -n thesis-avit` para adaptadores que lo requieran.
5. No instalar dependencias sin comprobar las versiones actuales.
6. Si se instala `segmentation-models-pytorch`, mantener `--no-deps`.
7. No volver a aplicar los parches antiguos si sus commits ya aparecen en `git log`.
8. Ante un `git am` fallido, no resolver a ciegas: inspeccionar y, si el cambio ya existe, usar `git am --abort`.
9. No reintroducir el límite de diez modelos.
10. No sobrescribir las máscaras crudas de Fase 0 al implementar FOV; conservar una comparación antes/después reproducible.
11. No confundir segmentación de lesión con diagnóstico de malignidad.
12. No llamar “checkpoint verificado” a un enlace visto en un README: hay que descargar, hashear, reconstruir, cargar y ejecutar.
13. Mantener visibles licencia, fuente, año y dataset del checkpoint.
14. Conservar resultados parciales cuando un modelo falle; un fallo no debe cancelar el lote completo.
15. Validar siempre que el PNG final sea binario y conserve el tamaño esperado.

---

## 18. Prompt sugerido para iniciar la fase siguiente

Copiar este texto junto con el presente archivo:

```text
Estoy continuando mi tesis en el repositorio Thesis_Fitzpatrick. Lee completamente el handoff de FASE 0: ESTADO INICIAL antes de responder. La Fase 0 cerró con 15 modelos, 38 pruebas y un lote equilibrado de 60 imágenes —10 por Fitzpatrick I–VI—, equivalente a 900 pares planificados. Los resultados crudos deben conservarse. Mi sistema es CachyOS con fish; el repositorio local está en /run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick. El servidor usa tesis-sam y los adaptadores pesados se ejecutan con conda run -n thesis-avit. No repitas el lote ni reinstales dependencias sin verificar el estado. La nueva fase debe tener su propio archivo Markdown independiente y no debe modificar el handoff de Fase 0.
```

---

## 19. Resumen mínimo de continuidad

```text
Repo: magicsistem/Thesis_Fitzpatrick
Ruta: /run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Thesis_Fitzpatrick
Shell: fish
Rama: agent/segmentation-review-ui
Base funcional auditada: a6e0abb
Rama remota: origin/agent/segmentation-review-ui
Entorno servidor: tesis-sam
Entorno adaptadores: thesis-avit
Piloto: 93 imágenes
Suite: 15 variantes
Muestra ejecutada: 60 imágenes, 10 por Fitzpatrick I–VI
Protocolo: 900 pares modelo-imagen
Inventario local: 904 directorios, 3375 archivos, 45 MB
Pruebas confirmadas: 38/38
Servidor: python scripts/serve_segmentation_review.py --host 127.0.0.1 --port 8000
Último arreglo confirmado: VM-UNet elimina total_ops/total_params raíz y anidados
Hito de cierre: fase-0-estado-inicial
Restricción: sin límite de 10; no sobrescribir resultados crudos; segmentación != malignidad
```
