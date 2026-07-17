# Búsqueda amplia de modelos para segmentación dermatoscópica

Fecha de corte de la búsqueda: **2026-07-16**.

## Objetivo

Identificar modelos con código y, preferentemente, checkpoints descargables para
segmentar la lesión completa en imágenes dermatoscópicas. El interés particular
es reducir dos fallos observados en el piloto:

1. máscaras agresivas que confunden el aro negro, las esquinas oscuras o el
   campo de visión circular del dermatoscopio con una lesión;
2. máscaras vacías o incompletas sobre lesiones benignas, pequeñas o de bajo
   contraste.

La búsqueda no asume que las métricas declaradas por los autores se reproducen
en la cohorte de esta tesis. Un modelo se considera candidato, no validado, hasta
que pase la misma inferencia, revisión visual y estratificación por Fitzpatrick
que los modelos ya integrados.

## Aclaración clínica y experimental

Una lesión **benigna también debe segmentarse** cuando existe una máscara de
lesión de referencia. La segmentación responde «¿qué píxeles pertenecen a la
lesión?», no «¿la lesión es maligna?». Una máscara vacía solo es correcta en una
imagen verdaderamente sin lesión anotada. Benignidad/malignidad debe conservarse
como estrato de auditoría, nunca como regla para borrar una máscara.

El aro circular oscuro es un artefacto óptico/campo de visión (FOV), no una
clase patológica. Trabajos de dermoscopia han tratado explícitamente el borde
negro y el viñeteado antes de calcular la lesión: [Watershed segmentation of
dermoscopy images](https://pmc.ncbi.nlm.nih.gov/articles/PMC3160671/) y
[Automatic Lesion Border Selection](https://pmc.ncbi.nlm.nih.gov/articles/PMC7173402/).
Por tanto, ningún checkpoint por sí solo puede garantizar que nunca lo marcará.
La solución más defendible es comparar mejores segmentadores **y** aplicar una
máscara independiente de FOV.

## Fuentes consultadas y método de búsqueda

Se cruzaron resultados de:

- PubMed/MEDLINE y PubMed Central para artículos biomédicos revisados por pares;
- IEEE Transactions on Medical Imaging e IEEE TCSVT mediante sus artículos,
  DOI y repositorios oficiales;
- MICCAI Open Access y CVPR/ICCV Open Access para trabajos de visión médica;
- Springer Nature, Scientific Reports, Sensors y Electronics para artículos
  abiertos;
- arXiv para preprints recientes, siempre contrastados con el repositorio del
  autor cuando existía;
- GitHub para código, licencia, releases, archivos de pesos y enlaces externos;
- Hugging Face Models y Spaces para tarjetas, licencias y archivos reales;
- Zenodo para registros de modelos y pesos con DOI.

Para cada candidato se comprobó en la fuente pública:

1. que la tarea fuera **máscara de lesión completa**, no clasificación, cajas o
   atributos dermatoscópicos;
2. que hubiese código suficiente para reconstruir el modelo;
3. que el checkpoint estuviese presente o enlazado por la fuente oficial;
4. qué licencia se declara, registrando también su ausencia sin ocultarla;
5. el dataset declarado y si el modelo requiere prompt, GPU o acceso aprobado;
6. si el diseño aporta algo relevante para bordes ambiguos, bajo contraste,
   artefactos o generalización.

Después de la búsqueda se ejecutó la auditoría local: se descargaron los pesos,
se fijaron revisiones y SHA-256, se inspeccionaron los estados y se escribieron
adaptadores. Los resultados definitivos, incluidos hallazgos que corrigieron la
tarjeta de algún modelo, están en `SEGMENTATION_CHECKPOINT_AUDIT.md`.

## Resultado ejecutivo

### Prioridad científica para el problema observado

| Prioridad | Modelo | Checkpoint público | Licencia | Prompt manual | Razón para probarlo |
|---:|---|---|---|---|---|
| 1 | De-LightSAM Dermoscopy | Sí, Google Drive oficial | Apache-2.0 | No se observa en el evaluador oficial | Entrenamiento desacoplado por modalidad y objetivo de generalización; checkpoint específico de dermoscopia |
| 2 | DermoSegDiff-A ISIC 2018 | Sí, SharePoint oficial | MIT | No | Modelo de difusión con pérdida consciente del borde; candidato directo para contornos ambiguos |
| 3 | VM-UNet ISIC 2017/2018 | Sí, Google Drive/Baidu oficiales | Apache-2.0 | No | Contexto global con SSM y dos checkpoints específicos de ISIC |
| 4 | U-Net/U-Net++ de Unixio | Sí, Hugging Face | MIT | No | Dos baselines con Dice autor-reportado cercano a 0.906 y código completo |
| 5 | SegFormer de Theodore Ioannidis | Sí, Hugging Face Space | MIT | No | Transformer compacto entrenado en ISIC 2018; útil como contraste arquitectónico |
| 6 | BiomedParse | Sí, pero con acceso aceptado | CC-BY-NC-SA-4.0 | Sí, texto | Foundation model con clase de dermoscopia; solo comparador condicionado a GPU y licencia no comercial |

La auditoría posterior descartó el H5 de DevBhuyan: internamente es un
clasificador ResNet50/densas sin salida espacial. En cambio, recuperó código y
licencia de Unixio y demostró que VM-UNet y De-LightSAM pueden tener una ruta
CPU de referencia, aunque previsiblemente lenta.

## Candidatos accesibles con checkpoint

### 1. De-LightSAM, variante Dermoscopy

- Artículo/código oficial:
  [xq141839/De-LightSAM](https://github.com/xq141839/De-LightSAM).
- Año: preprint 2024; aceptado en IEEE TCSVT en 2025 según el repositorio.
- Licencia: Apache-2.0.
- Checkpoint: el “Segmentation Model Zoo” enlaza un ZIP público de Google Drive
  y contiene una variante TinyViT específica de **Dermoscopy**.
- Dataset/formato observado: el repositorio usa imágenes y máscaras ISIC a
  1024 × 1024.
- Interfaz: el evaluador oficial llama `model(x=image, domain_seq=modalidad)`;
  no se observa un punto o caja dibujado por el usuario. `domain_seq` es la
  modalidad, no un prompt espacial.
- Coste: el entorno oficial es CUDA/PyTorch 1.13; el adaptador conserva la
  arquitectura y reemplaza las llamadas del evaluador por tensores CPU. La
  entrada 1024 × 1024 hace que siga siendo una prueba muy pesada.
- Interés: fue diseñado para segmentación médica generalizable por modalidad.
  No existe evidencia específica de que elimine el aro del dermatoscopio, pero
  sí es uno de los checkpoints más pertinentes para medir cambio de dominio.
- Estado: **implementado como experimento CPU no preseleccionado**.

### 2. DermoSegDiff-A y DermoSegDiff-B

- Artículo/código oficial:
  [xmindflow/DermoSegDiff](https://github.com/xmindflow/DermoSegDiff).
- Año: MICCAI PRIME 2023.
- Licencia: MIT.
- Checkpoints oficiales:
  - DermoSegDiff-A, entrenado para ISIC 2018;
  - DermoSegDiff-B, entrenado para PH2.
  Ambos se enlazan desde la tabla “Model weights” del README mediante
  SharePoint de la Universidad de Regensburg.
- Interfaz: automática, sin prompt manual.
- Interés: la arquitectura y la función de pérdida priorizan información de
  frontera, exactamente el tipo de señal que interesa para lesiones con borde
  difuso.
- Coste: los autores solicitan GPU de al menos 12 GB y reportan A100 de 80 GB
  para sus experimentos. La inferencia por difusión también puede ser más lenta
  que una U-Net.
- Riesgo: un mejor borde medio no prueba resistencia a esquinas oscuras; el FOV
  debe seguir siendo una evaluación separada.
- Estado propuesto: **alta prioridad en una máquina con GPU**, primero A/ISIC18;
  B/PH2 sirve para medir generalización cruzada.

### 3. VM-UNet ISIC 2017 e ISIC 2018

- Artículo/código oficial:
  [JCruan519/VM-UNet](https://github.com/JCruan519/VM-UNet).
- Año: 2024, posteriormente ACM TOMM.
- Licencia: Apache-2.0.
- Checkpoints: el README enlaza una
  [carpeta oficial de Google Drive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)
  y Baidu con modelos entrenados para ISIC 2017, ISIC 2018 y Synapse.
- Interfaz: automática, sin prompt.
- Interés: modela dependencias de largo alcance con estado espacial; aporta una
  familia distinta a AViT, BA-Transformer y U-Net.
- Coste: el entorno oficial usa `mamba_ssm`, `causal_conv1d`, Triton y CUDA. La
  integración sustituye ese kernel por la recurrencia PyTorch de referencia,
  sin modificar los pesos; será más lenta.
- Estado: **ambos checkpoints ISIC implementados y fijados por hash**.

### 4. Archivo H5 de DevBhuyan: descartado tras inspección

- Fuente:
  [DevBhuyan/Skin-Lesion-Segmentation](https://huggingface.co/DevBhuyan/Skin-Lesion-Segmentation).
- Año: la tarjeta no cita un artículo ni año de entrenamiento; el artefacto
  estaba público y actualizado al corte de 2026.
- Licencia: Apache-2.0.
- Framework anunciado: Keras/TensorFlow; la inspección real no encontró U-Net.
- Checkpoint: `2016_extend_best_model.h5`, aproximadamente 146 MB, presente en
  la rama principal.
- Dataset: la tarjeta declara ISIC 2016 e ISIC 2017 y métricas auto-reportadas;
  sin embargo, el árbol visible contiene solo el archivo nombrado para 2016.
  Esa discrepancia debe quedar registrada y no se debe presentar el H5 como dos
  checkpoints.
- Interfaz: automática, sin prompt.
- Hallazgo local: el H5 carece de `model_config` y contiene ResNet50, pooling,
  `flatten` y capas densas. Es un estado de clasificación, no una máscara.
- Estado: **excluido**; no se implementa una salida ficticia a partir de logits
  de clasificación.

### 5. Space de Theodore Ioannidis: SegFormer, U-Net e Inception

- Fuente:
  [theodore-ioann/Skin-Lesion-Segmentation](https://huggingface.co/spaces/theodore-ioann/Skin-Lesion-Segmentation/tree/main).
- Año: 2025.
- Checkpoints presentes:
  - `segformer.pt`, 14.9 MB;
  - `unet.pt`, 30.9 MB;
  - `inception.pt`, 14.4 MB.
- Código: `supervised.py` reconstruye las tres arquitecturas; SegFormer usa
  `nvidia/segformer-b0-finetuned-ade-512-512` con dos clases.
- Dataset: el README identifica ISIC 2018 Task 1 y describe evaluación de
  segmentación de lesión.
- Licencia: MIT, confirmada por la API del Space.
- Interfaz: automática, sin prompt.
- Riesgo: es un proyecto comunitario sin tarjeta formal; la ausencia de licencia
  impide redistribuir código/pesos dentro del repositorio hasta obtener permiso.
- Estado: **U-Net, Inception y SegFormer integrados**, con revisión y hashes
  fijados.

### 6. Repositorio Hugging Face de Unixio

- Fuente:
  [unixio/unet-skin-lesion-segmentation](https://huggingface.co/unixio/unet-skin-lesion-segmentation/tree/main).
- Año: 2026.
- Checkpoints presentes:
  - `best_attention_unet.pt`, 126 MB;
  - `best_unet.pt`, 97.9 MB;
  - `best_unetpp.pt`, 105 MB.
- Licencia: MIT, confirmada en el Space asociado.
- Código: el Space enlaza el repositorio público de BertinAm y contiene las
  clases exactas, normalización ImageNet, entrada 256 × 256 y configuración.
- Estado: **U-Net, U-Net++ y Attention U-Net integrados**, con los tres hashes
  fijados.

### 7. BiomedParse

- Artículo/modelo oficial:
  [microsoft/BiomedParse](https://huggingface.co/microsoft/BiomedParse), Nature
  Methods 2025.
- Licencia: CC-BY-NC-SA-4.0; uso de investigación, no clínico.
- Acceso: repositorio público condicionado a iniciar sesión, aceptar términos y
  compartir información de contacto.
- Interfaz: requiere prompt textual. La tarjeta recomienda para dermoscopia
  `skin: lesion` y `melanoma`.
- Entrada/coste: 1024 × 1024, Detectron2 y configuración CUDA en el ejemplo
  oficial; batch 1.
- Interés: comparador de foundation model multimodal y posible fuente de
  desacuerdo/incertidumbre.
- Limitaciones: no es directamente comparable con los modelos automáticos sin
  prompt, la licencia es no comercial y el propio autor advierte sobre cambio de
  distribución externa.
- Estado propuesto: **comparador opcional**, no siguiente modelo CPU ni parte
  del ranking principal sin una categoría separada “con prompt”.

## Modelos prometedores sin checkpoint actualmente verificable

Estos modelos son relevantes para seguimiento, pero no deben aparecer como
ejecutables en la interfaz actual.

| Modelo | Evidencia relevante | Código | Motivo de espera |
|---|---|---|---|
| LB-UNet | MICCAI 2024; asistencia explícita de frontera y 38 KB de parámetros | [Repositorio](https://github.com/xuxuxuxuxuxjh/LB-UNet) | El artículo dice que hay modelos entrenados, pero el árbol actual solo expone código/datos de frontera; sin release ni pesos visibles y sin licencia declarada |
| autoSMIM | IEEE TMI 2023; preentrenamiento auto-supervisado para bordes borrosos | [Repositorio](https://github.com/Wzhjerry/autoSMIM) | Código de entrenamiento completo, pero sin releases/checkpoints finales; entorno de 4 RTX 3090 |
| SkinFormer 2024 | Transformer de textura estadística; Dice declarado 93.2% en ISIC18 | [Repositorio](https://github.com/Rongtao-Xu/SkinFormer) | Repositorio parcial, sin pesos, sin licencia y marcado “under review” |
| ScaleFusionNet | Fusión CNN/Transformer y refinamiento de frontera; evaluación ISIC16/18 y PH2 | [Repositorio](https://github.com/sqbqamar/ScaleFusionNet) | Código de entrenamiento/predicción, pero no hay release, pesos ni licencia visible |
| VML-UNet | 0.53 M parámetros; atención a bajo contraste, ruido y fondos complejos | [Artículo abierto](https://www.mdpi.com/2079-9292/14/14/2866) | El artículo no proporciona repositorio/checkpoint reproducible |
| ELA-Net | 0.459 M parámetros; atención ligera, bordes borrosos e irregulares | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39001081/) | Artículo abierto, pero no se localizó código/peso oficial |
| SkinFormer 2026 | Híbrido ViT/ConvNeXtV2 para bajo contraste y artefactos | [Scientific Reports](https://www.nature.com/articles/s41598-026-48633-w) | Trabajo distinto al SkinFormer 2024; no se localizó checkpoint oficial |
| DSNet | El artículo declara robustez a pelo/artefactos y pesos públicos | [Artículo](https://doi.org/10.1016/j.compbiomed.2020.103738) | La auditoría local existente no pudo recuperar pesos actuales del repositorio citado |

Las páginas biomédicas que respaldan específicamente autoSMIM y ScaleFusionNet son
[autoSMIM en PubMed](https://pubmed.ncbi.nlm.nih.gov/37379178/) y
[ScaleFusionNet en PubMed](https://pubmed.ncbi.nlm.nih.gov/41038982/).

## Recursos útiles que no son un segmentador implementable

### FEDD/sDDI para evaluar piel diversa y artefactos

El repositorio [hectorcarrion/FEDD](https://github.com/hectorcarrion/fedd)
publica máscaras sDDI con etiquetas separadas para **lesión, marcador, regla y
piel**, además de particiones por tono de piel. No ofrece un checkpoint final
listo para esta web, pero es muy valioso para diseñar pruebas que midan si el
segmentador invade reglas o marcadores y para no depender solo de ISIC.

### Benchmark de dominios por artefacto

El repositorio
[EPVT-and-Skin-DG-benchmark](https://github.com/SiyuanYan1/EPVT-and-Skin-DG-benchamrk)
organiza imágenes por `clean`, `dark_corner`, `gel_bubble` y otros artefactos.
Su tarea principal es reconocimiento/clasificación, no máscara de lesión, pero
su agrupación puede reutilizarse como índice de auditoría del conjunto ISIC.

### Zenodo

La búsqueda en Zenodo no encontró un checkpoint final de segmentación
dermatoscópica con código y procedencia comparable a los anteriores. Sí encontró:

- [pesos de encoders genéricos de segmentación médica](https://zenodo.org/records/13971513),
  4.3 GB, útiles para reentrenar pero no una máscara de lesión lista;
- [un trabajo TMU-Net/DARTS](https://zenodo.org/records/20235411) cuyo registro
  contiene únicamente el PDF;
- modelos ISIC de **clasificación**, por ejemplo
  [Skin Lesion Images for Melanoma Classification](https://zenodo.org/records/7716488),
  que no deben confundirse con segmentación.

## Exclusiones importantes

- El modelo
  [YOLOv11s ISIC 2018](https://huggingface.co/raj5517/yolov11s-skin-lesion-isic2018)
  es detección/clasificación de Task 3 y usa cajas aproximadas; no produce la
  frontera de Task 1.
- [chvlyl/ISIC2018](https://github.com/chvlyl/ISIC2018) segmenta atributos
  internos (red de pigmento, glóbulos, estrías, etc.), no la lesión completa.
- Checkpoints de ImageNet, SAM genérico o encoders preentrenados no cuentan como
  modelos finales de lesión hasta que exista fine-tuning verificable.
- Un artículo con código, pero sin checkpoint, permanece en seguimiento; no se
  rellena el hueco con forks de procedencia incierta.

## Protocolo recomendado contra segmentaciones agresivas

### 1. Máscara independiente de campo de visión

Generar `valid_fov_mask` a partir del borde/viñeteado, conservando tres archivos:

1. imagen original sin alterar;
2. máscara de FOV válida;
3. predicción del modelo reproyectada al tamaño original y limitada al FOV.

No conviene rellenar siempre el aro con “color medio de piel” antes de inferir:
puede crear una textura artificial y contaminar precisamente la media que luego
se usará para Fitzpatrick. Es preferible recortar/reproyectar o enmascarar fuera
del FOV, y calcular el color solo en píxeles de piel válida.

### 2. No eliminar automáticamente lesiones que tocan el borde

El contacto con el límite del FOV debe producir una **bandera de control de
calidad**, no un descarte automático: una lesión real puede estar cortada por el
encuadre. Registrar:

- porcentaje de píxeles positivos fuera del FOV;
- fracción del perímetro predicho que toca el borde válido;
- cobertura total de la máscara;
- número y tamaño de componentes conectados;
- máscara vacía, máscara casi completa y piel limpia insuficiente.

### 3. Conjunto de prueba estratificado

Además de Fitzpatrick I–VI, construir estratos no excluyentes:

- aro negro/esquinas oscuras/viñeteado;
- pelo denso;
- reflejos, gel y burbujas;
- regla, tinta o parche;
- lesión pequeña;
- lesión de bajo contraste;
- benigno y maligno, solo para auditar disparidad;
- imagen sin lesión de referencia, si el dataset realmente contiene esa clase.

La búsqueda histórica de ISIC recuerda que incluso los mejores métodos del reto
2018 fallaban en más del 10% de las imágenes y que igual rendimiento interno no
implicaba igual generalización: [ISIC 2018 Challenge
paper](https://arxiv.org/abs/1902.03368).

### 4. Métricas de selección

No escoger el “mejor” solo por Dice promedio. Para cada modelo y estrato medir:

- Dice e IoU;
- Boundary F1 y HD95;
- falso positivo fuera del FOV;
- tasa de máscara vacía y casi completa;
- tasa de fallo crítico por revisión humana;
- número de píxeles de piel limpia disponibles para el cálculo cromático;
- diferencia de desempeño por Fitzpatrick y por benigno/maligno;
- tiempo por imagen, CPU/GPU, núcleos efectivos y RAM/VRAM máxima.

La decisión debe priorizar primero la menor tasa de fallo crítico, después el
desempeño de frontera y finalmente el promedio global. Un ensemble o selector
de máscaras puede ser superior a confiar siempre en un único modelo; un trabajo
clásico obtuvo mejores bordes seleccionando entre varios algoritmos que usando
el mejor individual en todas las imágenes:
[Automatic Lesion Border Selection](https://pubmed.ncbi.nlm.nih.gov/30868667/).

## Orden de implementación propuesto

1. **Unixio U-Net, U-Net++ y Attention U-Net**: implementados.
2. **Theodore U-Net, Inception y SegFormer**: implementados.
3. **VM-UNet ISIC 2017 e ISIC 2018**: implementados con selective scan CPU.
4. **De-LightSAM Dermoscopy**: implementado como prueba CPU pesada.
5. **DermoSegDiff-A**: pendiente porque el SharePoint oficial exige acceso.
6. **BiomedParse**: comparador futuro separado por requerir prompt y aceptación
   de términos.

Por decisión experimental, la corrección de FOV queda fuera de esta fase: se
comparan primero las máscaras crudas de cada checkpoint.

## Criterio de admisión final a la web

Un candidato nuevo entra en `configs/segmentation_models.json` solamente cuando:

1. el checkpoint se descarga desde una URL trazable y se fija su SHA-256;
2. código suficiente permite reconstruir la arquitectura y la licencia —o su
   ausencia— queda visible para decidir el alcance del uso;
3. el `state_dict` carga estrictamente o se documenta cada transformación;
4. una imagen real genera una máscara binaria del tamaño original;
5. una máscara vacía o completa se conserva como resultado revisable, sin romper
   el lote;
6. se ejecuta primero el piloto estratificado sobre la máscara cruda; la
   corrección de FOV se evaluará en una fase posterior separada;
7. la interfaz muestra procedencia, año, licencia, tiempos y memoria;
8. la revisión no detecta regresiones críticas frente a los modelos actuales.

Este documento es una lista de candidatos para investigación, no una afirmación
de eficacia clínica ni una autorización para diagnóstico.
