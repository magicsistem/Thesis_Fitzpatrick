# Auditoría reproducible de checkpoints de segmentación

Fecha de verificación: **2026-07-16**.

## Objetivo y regla de inclusión

Esta auditoría revisa los modelos del informe de investigación y los baselines
añadidos durante la implementación. Un modelo aparece en la interfaz solamente
si cumple todas estas condiciones:

1. el repositorio o la fuente del modelo sigue disponible;
2. existe un checkpoint ya entrenado para segmentación de lesiones cutáneas;
3. el archivo puede descargarse sin credenciales ni contacto privado;
4. el archivo no es un marcador vacío y su SHA-256 puede fijarse;
5. su arquitectura puede reconstruirse y el estado puede cargarse de forma
   estricta;
6. existe un adaptador automático que produce una máscara binaria en CPU.

Un enlace escrito en un README no cuenta como checkpoint verificado si la
descarga ya no funciona. Un modelo que solo ofrece código de entrenamiento no
entra en la web. Los modelos descartados siguen documentados aquí para que la
búsqueda no tenga que repetirse ni se confunda “publicado en un artículo” con
“checkpoint público y reproducible”.

## Procedimiento aplicado a cada candidato

Para cada modelo se hicieron, según correspondía, estas comprobaciones:

```bash
gh repo view OWNER/REPO --json nameWithOwner,url,defaultBranchRef,isArchived,isPrivate
git ls-remote https://github.com/OWNER/REPO.git HEAD
git clone --depth 1 https://github.com/OWNER/REPO.git /tmp/REPO
find /tmp/REPO -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.pkl' \
  -o -iname '*.ckpt' -o -iname '*.h5' -o -iname '*.weights' \)
rg -n -i 'pretrain|checkpoint|weight|drive.google|baidu|download' \
  /tmp/REPO --glob 'README*' --glob '*.md'
```

Cuando apareció un enlace, se intentó la descarga con `gdown` o mediante la
URL inmutable del proveedor. Después se inspeccionó el archivo con `file`,
`unzip -l`, tamaño y `sha256sum`. En los modelos integrados también se verificó
el commit del código, carga estricta con `load_state_dict(..., strict=True)`,
preprocesamiento oficial, salida binaria y una inferencia real antes de
conectarlos a la web.

## Resultado resumido

| Modelo o variante | Checkpoint utilizable | Estado | Motivo |
|---|---:|---|---|
| AViT ISIC | Sí | Implementado | Descarga, hash, carga estricta e inferencia verificadas |
| UltraLight VM-UNet | Sí | Implementado | Checkpoint pequeño verificado; dataset exacto no indicado por el autor |
| BA-Transformer ISIC 2016 | Sí | Implementado | ZIP y checkpoint interno verificados |
| U-Net/ResNet34 ISIC 2018 | Sí | Implementado | Revisión inmutable y hash verificados |
| SkinMamba ISIC 2017 | Sí | Implementado | Checkpoint oficial verificado; selective scan adaptado a CPU |
| SkinMamba ISIC 2018 | Sí | Implementado | Checkpoint oficial verificado; selective scan adaptado a CPU |
| Unixio U-Net ISIC 2018 | Sí | Implementado | Código MIT, revisión HF y hash verificados |
| Unixio U-Net++ ISIC 2018 | Sí | Implementado | Código MIT, revisión HF y hash verificados |
| Unixio Attention U-Net ISIC 2018 | Sí | Implementado | Arquitectura pública y estado `model_state` verificados |
| Theodore U-Net ISIC 2018 | Sí | Implementado | Space MIT fijado y checkpoint verificado |
| Theodore Inception ISIC 2018 | Sí | Implementado | Space MIT fijado y checkpoint verificado |
| Theodore SegFormer-B0 ISIC 2018 | Sí | Implementado | Space MIT fijado y checkpoint verificado |
| VM-UNet ISIC 2017 | Sí | Implementado | Checkpoint oficial y ruta selective-scan CPU |
| VM-UNet ISIC 2018 | Sí | Implementado | Checkpoint oficial y ruta selective-scan CPU |
| De-LightSAM Dermoscopy | Sí | Implementado | ZIP oficial, checkpoint ISIC y commit verificados |
| DermoSegDiff-A/B | No recuperable hoy | No implementado | SharePoint oficial redirige a una respuesta 401 |
| DevBhuyan `2016_extend_best_model.h5` | No es segmentador | No implementado | H5 de solo pesos para clasificador ResNet50/densas |
| BiomedParse | Acceso condicionado | No implementado | Requiere aceptar términos e iniciar sesión |
| ISCF ISIC 2017/2018 | No recuperable | No implementado | Los dos enlaces oficiales rechazaron la descarga |
| TMUNet | No verificable | No implementado | Repositorio original ya no existe |
| Attention DeepLabv3+ | No verificable | No implementado | Repositorio original ya no existe |
| BCDU-Net | No verificable | No implementado | Repositorio original ya no existe |
| Superpixel Merging | No aplica | No implementado | Método clásico sin checkpoint neuronal |
| DSNet | No | No implementado | Código/notebook sin pesos publicados |
| MALUNet | No | No implementado | Solo dataset y entrenamiento |
| EGE-UNet | No | No implementado | Solo dataset y entrenamiento |
| UCM-Net | No | No implementado | Solo código de entrenamiento |
| MDViT | No | No implementado | No se publicaron pesos en repo/README |
| MHorUNet | No | No implementado | `test.py` exige una ruta creada por entrenamiento |
| HSH-UNet | No | No implementado | `test.py` exige una ruta creada por entrenamiento |
| SET | No | No implementado | El README mantiene la publicación de pesos como tarea futura |
| EM-Net | No público | No implementado | Archivos `.pth` de 1 byte; pesos reales solo por correo |
| MobileSAM | Sí, local | No integrado en este benchmark | Requiere prompts y selección de candidatos; no es comparable automáticamente |

## AViT (implementado)

- Fuente: <https://github.com/siyi-wind/AViT>.
- Se confirmó el repositorio con `gh repo view`, se clonó y se fijó el commit
  `b77b01af263727b2b2cf2555a5ed9f1f48a2b4a2`.
- Se inspeccionó el README y su carpeta pública de Google Drive. Se descargaron
  los pesos publicados y se seleccionó `AViT_ViT_B_ISIC_best.pth`, no los
  checkpoints de Dermofit, SCD ni PH2.
- Tamaño observado: aproximadamente 380 MiB.
- SHA-256:
  `9b4ad401483d96535769433f4781da42179bec6a7ef932a1d02a7f786e9f24db`.
- El adaptador reconstruye la arquitectura oficial, reproduce la normalización
  ImageNet de 224 × 224, carga estrictamente el estado y restaura la máscara al
  tamaño original. Se ejecutó una imagen real y se obtuvo una máscara binaria.
- Decisión: **incluido en la web** como modelo normal/pesado.

## UltraLight VM-UNet (implementado)

- Fuente: <https://github.com/wurenkai/UltraLight-VM-UNet>.
- Se verificó el repositorio, se siguió el enlace de pesos publicado por el
  autor y se fijó el commit
  `27e44181b3cd5b0b2ab3ad1e3ddb8cad67368fbd`.
- ZIP SHA-256:
  `ba51198d93f70c0281ca81d1f466c937f449b81b5fe30775a4a2bd8b0e379c99`.
- Checkpoint `UltraLight_VM_UNet.pth`, aproximadamente 232 KiB, SHA-256:
  `43b11155c19c2296707ec4dee3c417529ea0b54eb111adee864b2274ec8df52a`.
- Se inspeccionaron nombres y formas del estado, se reprodujo la normalización
  min-max del repositorio y se sustituyó el kernel Mamba CUDA por la misma
  recurrencia escrita en PyTorch CPU, conservando parámetros y carga estricta.
- Se ejecutó una imagen real; la salida tuvo únicamente valores 0 y 255.
- Limitación preservada: el autor llama al archivo “skin lesion weights”, pero
  no identifica cuál conjunto produjo ese checkpoint.
- Decisión: **incluido en la web** como modelo ligero.

## BA-Transformer (implementado)

- Fuente: <https://github.com/jcwang123/BA-Transformer>.
- Se fijó el commit `d1ccdff68beac82d18c2cd64bb800e51d7afc5e1`.
- Se siguió el enlace de pesos del README y se descargó el ZIP de Google Drive.
  ZIP SHA-256:
  `0d837469bfbc7306d1fc61d1f4292243d4447a0f45d2ddc850f7947932aec88e`.
- `unzip -l` mostró directorios `isic2016` aunque el texto del enlace decía
  PH2. Se extrajo
  `isic2016/bat_1_1_0_e6_loss_0_aug_1/fold_0/model/best.pkl`.
- Checkpoint final, aproximadamente 179 MiB, SHA-256:
  `62b4148b26b01b0b17b4125d74115ed49c507eab4d39ec8ff8e063a2f7980233`.
- La primera ejecución reveló una ruta absoluta del autor para ResNet-50. Se
  aisló esa carga de arranque y después se cargó el checkpoint BA completo con
  `strict=True`, eliminando únicamente el prefijo multi-GPU `module.`.
- Se verificó inferencia real en CPU. Una máscara completamente negra es una
  predicción válida cuando el modelo no detecta lesión, no un error de carga.
- Decisión: **incluido en la web** como modelo normal/pesado.

## U-Net/ResNet34 ISIC 2018 (implementado)

- Fuente de entrenamiento:
  <https://github.com/Jesusrodriguezf90/skin-lesion-analysis>.
- Fuente del checkpoint:
  <https://huggingface.co/Jesusrodriguezf90/unet-resnet34-isic2018-segmentation>.
- Se inspeccionó la tarjeta del modelo, arquitectura, preprocesamiento y dataset.
  La descarga quedó fijada a la revisión inmutable
  `03d64aeb97ef50ae1b3a68224645c7eeb081a61a`.
- Checkpoint `best_unet_resnet34.pth`, aproximadamente 94 MiB, SHA-256:
  `1ea87e341552768234b367c3b68704030bc1bc08991c323508ea5c5086d9d334`.
- Se reconstruyó U-Net/ResNet34 con
  `segmentation-models-pytorch==0.5.0`, normalización ImageNet a 256 × 256 y
  carga estricta. Se verificó inferencia binaria real.
- Es un baseline comunitario reproducible, no una arquitectura nueva revisada
  por pares; esa limitación aparece también en la tarjeta web.
- Decisión: **incluido en la web** como baseline normal.

## SkinMamba ISIC 2017 e ISIC 2018 (implementado)

- Fuente: <https://github.com/zs1314/SkinMamba>.
- Se confirmó licencia Apache-2.0, se clonó y se fijó el commit
  `131a85da14da4d3bf135b52912d274191c05f3b8`.
- El README ofrece Baidu y Google Drive. Se probó el ID de Google Drive
  `1ialdv8WJoKEZkkwiqGuleEWnXOyyUEoW` con `gdown`; la descarga de 105 MB fue
  satisfactoria.
- ZIP SHA-256:
  `95cda03c7952a3c7a984dbd4403ab7f821ff10277939554f52d49a7fe28fb4b0`.
- `unzip -l` confirmó dos pesos reales de unos 54 MiB cada uno:
  - `weight/isic17.pth`:
    `9b941f1459dd6e7660bd5ec32b2b58ba861c596ac8b14708a43a9c940f0645a7`;
  - `weight/isic18.pth`:
    `36a2c352cd39011db3f416373be1bdd98b079d031fc778dbc640780823388543`.
- Se inspeccionó el contenedor PyTorch. Es un `OrderedDict` real; también guarda
  buffers de perfilado THOP (`total_ops`, `total_params`), que el adaptador
  elimina antes de la carga estricta porque no son parámetros del modelo.
- El código oficial exige una extensión CUDA. El propio repositorio incluye
  la formulación PyTorch del selective scan para CPU/GPU. El adaptador usa esa
  recurrencia por bloques, reproduce las medias/desviaciones oficiales de test
  y el tamaño 224 × 224, sin cambiar los pesos aprendidos.
- Se presentan como dos variantes separadas para que la revisión pueda comparar
  el efecto de entrenar en ISIC 2017 frente a ISIC 2018.
- Decisión: **ambos checkpoints incluidos en la web**.

## Unixio/BertinAm: U-Net, U-Net++ y Attention U-Net (implementados)

- Fuentes:
  <https://github.com/BertinAm/unet-skin-lesion-segmentation> y
  <https://huggingface.co/unixio/unet-skin-lesion-segmentation>.
- Se comprobó además el Space asociado, revisión
  `c6c8c0d1e588cd9966ff0bdffcfbe54d2e9ccd2d`, que publica el código de
  inferencia, configuración 256 × 256, normalización ImageNet y licencia MIT.
- Los pesos se descargaron desde la revisión inmutable del modelo
  `2a8b59e2de58d87370a92956cdea250ddf95531d`:
  - `best_unet.pt`, 97,921,547 bytes, SHA-256
    `99bf0a32d36ac28a42db1299af6a0ad4d562efdf685cf3fcdccd3ae8f94d1448`;
  - `best_unetpp.pt`, 104,526,675 bytes, SHA-256
    `d2ba5876ffd449f426205d8498b1165618ee0159a286bfa701ff2492544ca269`;
  - `best_attention_unet.pt`, 125,682,667 bytes, SHA-256
    `0d41ac498ac44d2b07ea88ef75f92370ec5312d7e6d32c39f9f0a0ce12b4812b`.
- `torch.load` mostró un diccionario con `model_state` y `config`. U-Net y
  U-Net++ se reconstruyen con `segmentation-models-pytorch==0.5.0`, encoder
  ResNet34 sin volver a descargar ImageNet. Attention U-Net reproduce las
  cuatro compuertas y nombres de módulos del código publicado.
- El adaptador interpola el mapa de probabilidad al tamaño original antes del
  umbral 0.5, igual que el predictor oficial. No aplica limpieza morfológica.
- Decisión: **las tres variantes se incluyen en la web**.

## Theodore Ioannidis: U-Net, Inception y SegFormer (implementados)

- Fuente: <https://huggingface.co/spaces/theodore-ioann/Skin-Lesion-Segmentation>.
- La API de Hugging Face confirmó licencia MIT. Se fijó la revisión completa
  `171a6dc86ee73faae0fdbda0f157d92bdb1c6596` y se inspeccionaron
  `supervised.py`, `utils.py` y el flujo real de la aplicación.
- Checkpoints:
  - `unet.pt`, 30,861,979 bytes, SHA-256
    `d58601e6433b7ebeab5ec4249ca02e413b62b28cd9e69b1719a368cb6deae5bb`;
  - `inception.pt`, 14,355,657 bytes, SHA-256
    `18c17a1d87be5906b31a76558132e3c3fc16e643747b8e0859c25cb914eadce9`;
  - `segformer.pt`, 14,946,661 bytes, SHA-256
    `0dcb4e5c9d19ab4caffaa324e4cf26090bcc9d37fb47840e38d81268691d8341`.
- Las tres redes producen dos clases a 128 × 128 y la lesión es la clase 1 por
  `argmax`. U-Net e Inception reciben tensor RGB 0–1; SegFormer añade la
  normalización ImageNet de su procesador oficial. El adaptador no descarga el
  backbone ADE en inferencia: construye la configuración B0 y carga el estado
  final completo con `strict=True`.
- Decisión: **las tres variantes se incluyen en la web**.

## VM-UNet ISIC 2017 e ISIC 2018 (implementados)

- Fuente: <https://github.com/JCruan519/VM-UNet>, licencia Apache-2.0 y commit
  fijado `b87827eb5a5faff00bd4b7b6505e56d3ca813ca2`.
- Se enumeró la carpeta Google Drive del README y se descargaron los dos
  archivos finales, no los `.whl` CUDA ni los encoders genéricos:
  - `best-vmunet-isic17.pth`, 109,836,533 bytes, SHA-256
    `61af065274210838ed65d165bd908a0b831ec6be8f2739c60e990fcfdd1071ce`;
  - `best-vmunet-isic18.pth`, 109,836,533 bytes, SHA-256
    `a5b2c175ccb2e2fa428004a1c90023ffd80ef3a9d9c485f7f77ccbe4427abd38`.
- La inspección del contenedor confirmó parámetros `vmunet.*` y buffers THOP
  `total_ops`/`total_params`. Solo esos buffers de perfilado y un posible
  prefijo `module.` se retiran antes de la carga estricta.
- Se reprodujeron la configuración `[2,2,2,2]`, decoder `[2,2,2,1]`, entrada
  256 × 256 y normalización min-max oficial. El kernel CUDA de VMamba se
  reemplaza por la recurrencia selective-scan equivalente escrita en PyTorch.
  Esta ruta conserva los pesos y admite CPU, pero puede ser muy lenta.
- Decisión: **ambos checkpoints se incluyen como variantes separadas**.

## De-LightSAM Dermoscopy (implementado)

- Fuente: <https://github.com/xq141839/De-LightSAM>, licencia Apache-2.0 y
  commit fijado `d2260d75e4ad1f5da8e053711cd9c36128088a7b`.
- Se descargó el ZIP oficial de Google Drive, 308,302,182 bytes, SHA-256
  `837e3f4654abe849aa634be2da3864b09d3b2a22f16b8178815afede187b0c9f`.
  `unzip -l` mostró seis modalidades. Se extrajo únicamente
  `ESP-MedSAM/ESP_isic_best.pth`, 51,383,697 bytes, SHA-256
  `79555730abffb5b39bef5d59afb7b89b13468348b77efa144649b90fddc01778`.
- El evaluador selecciona dermoscopia con `domain_seq=0`; no requiere punto ni
  caja manual. El adaptador elimina las llamadas `.cuda()` del evaluador, no
  del modelo, y ejecuta TinyViT/decoder en CPU a 1024 × 1024.
- Se preservó incluso la normalización efectiva del `BinaryLoader` oficial,
  que por error usa `pixel_mean` también como divisor. Cambiarla habría
  alterado la distribución con la que se entrenó el checkpoint.
- Decisión: **incluido como experimento automático muy pesado en CPU**.

## DermoSegDiff (no implementado por descarga bloqueada)

- Fuente: <https://github.com/xmindflow/DermoSegDiff>, licencia MIT.
- El README enlaza carpetas SharePoint para DermoSegDiff-A/ISIC18 y B/PH2. Se
  siguió la redirección del enlace público oficial; el destino respondió HTTP
  401 y exigió autenticación. Sin bytes no se puede fijar hash ni verificar la
  carga.
- Decisión: **no se muestra como ejecutable** hasta que el autor restaure una
  descarga anónima o publique los archivos en un repositorio versionado.

## DevBhuyan `2016_extend_best_model.h5` (excluido: no es segmentador)

- Fuente: <https://huggingface.co/DevBhuyan/Skin-Lesion-Segmentation>, revisión
  `d2cff8f1edd7aef351ce5967e901c2e190a57c7e`.
- Se descargó el H5 de 146,031,800 bytes; SHA-256
  `598edaf686f3334adfa71b17ca2993fd70df88c4e6a2d27a31d24dd03294b57d`.
- La inspección HDF5 reveló `keras_version=2.11.0`, ausencia de `model_config`
  y capas `resnet50`, `flatten`, global pooling y varias `dense`. No hay decoder
  U-Net ni salida espacial; es un archivo de solo pesos para clasificación.
- Decisión: **no implementado**, aunque la tarjeta lo denomine segmentación.

## ISCF (no implementado)

- Fuente: <https://github.com/saniaesk/skin-lesion-segmentation>, commit auditado
  `2618f7d707e3af7ac7fb7eec841207ce5536e4d9`.
- Se localizaron en el README los IDs oficiales:
  - ISIC 2017: `1T-cvswKf1slTqEeXylbKQ-m6KWfRUsJI`;
  - ISIC 2018: `1vVR7iqamGprzO9BXO4GzOrejpGdr4p4B`.
- Se intentaron ambos con `gdown` 5.2.0 y 6.1.0. Los dos respondieron que no se
  podía recuperar el enlace público. También se intentó el endpoint directo de
  Google Drive y respondió 404.
- Sin archivo no es posible calcular hash, inspeccionar las claves ni probar
  carga estricta. No se sustituyó el peso por otro archivo no oficial.
- Decisión: **eliminado de la web**; checkpoint anunciado pero no recuperable.

## TMUNet (no implementado)

- URL investigada: <https://github.com/rezazad68/TMUnet>.
- `gh repo view rezazad68/TMUnet` devolvió “Could not resolve to a Repository”.
  `git ls-remote` no devolvió HEAD y `git clone --depth 1` respondió
  “Repository not found”.
- Al no existir la fuente original, tampoco se pueden validar los antiguos
  enlaces de pesos citados por el informe ni reconstruir exactamente el modelo.
- Decisión: **eliminado de la web**.

## Attention DeepLabv3+ (no implementado)

- URL investigada: <https://github.com/rezazad68/AttentionDeeplabv3p>.
- Se repitieron `gh repo view` y `git ls-remote`; GitHub no resolvió el
  repositorio. No se adoptaron forks sin procedencia como reemplazo del oficial.
- Decisión: **eliminado de la web**.

## BCDU-Net (no implementado)

- URL investigada: <https://github.com/rezazad68/BCDU-Net>.
- `gh repo view` no resolvió el repo y `git ls-remote` no produjo un HEAD. Los
  pesos mencionados en referencias antiguas no pueden vincularse hoy a código
  oficial verificable.
- Decisión: **eliminado de la web**.

## Superpixel Merging (no implementado)

- Fuente: <https://github.com/dipaco/superpixel-skin-lesion-segmentation>, HEAD
  `18054434749d348419c84004cbfdb197d99f4bef`.
- Se clonó, se buscaron extensiones de pesos y términos de checkpoint/descarga.
  No aparecieron pesos porque es un método clásico basado en superpíxeles, no
  una red con checkpoint.
- Decisión: **no entra en un catálogo condicionado a checkpoint**.

## DSNet (no implementado)

- Fuente:
  <https://github.com/kamruleee51/Skin-Lesion-Segmentation-Using-Proposed-DSNet>,
  HEAD `e814c77762568cfb678acfde757b1913de1adc5d`.
- Se clonó y se recorrieron README, notebooks y extensiones `.pth`, `.pt`,
  `.h5`, `.ckpt` y `.weights`. No existe checkpoint descargable ni enlace de
  pesos; el repositorio proporciona el experimento para entrenar.
- Decisión: **eliminado de la web**.

## MALUNet (no implementado)

- Fuente: <https://github.com/JCruan519/MALUNet>, HEAD
  `e184b47de99b3fda6b7880fcf3ff99b806308d2b`.
- Se buscaron archivos de pesos y enlaces. El Google Drive/Baidu del README con
  ID `1XM10fmAXndVLtXWOt5G0puYSQyI2veWy` corresponde al **dataset dividido**, no
  a pesos. No hay checkpoint en el árbol Git ni enlace de modelo preentrenado.
- Decisión: **eliminado de la web**.

## EGE-UNet (no implementado)

- Fuente: <https://github.com/JCruan519/EGE-UNet>, HEAD
  `f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a`.
- Se clonó y se recorrieron código, README y extensiones de pesos. El único
  enlace grande es el mismo dataset ISIC 17/18 dividido; no es checkpoint.
- Decisión: **eliminado de la web**.

## UCM-Net (no implementado)

- Fuente: <https://github.com/chunyuyuan/UCM-Net>, HEAD
  `de98f4298b0518cdc6b321dbff4f2bb69ab9b332`.
- Se buscaron archivos, releases y términos de descarga/pretrained/checkpoint.
  El repositorio contiene arquitectura y entrenamiento, pero ningún estado
  entrenado público.
- Decisión: **eliminado de la web**.

## MDViT (no implementado)

- Fuente: <https://github.com/siyi-wind/MDViT>, HEAD
  `6ecc4023ae64ef32d8265784c205842329a8b8cf`.
- Se clonó y se inspeccionaron README, carpetas y extensiones de checkpoint. No
  existe archivo entrenado ni enlace público a pesos; solo código experimental.
- Decisión: **eliminado de la web**.

## MHorUNet (no implementado)

- Fuente: <https://github.com/wurenkai/MHorUNet>, HEAD
  `d8d9a2e9f6cbb68fdbf83d6931f2ceba7e7688b3`.
- No apareció ningún archivo de pesos ni enlace de descarga. El README indica
  que el usuario debe editar `resume_model` en `test.py` para apuntar al
  checkpoint producido por su propio entrenamiento.
- Decisión: **eliminado de la web**.

## HSH-UNet (no implementado)

- Fuente: <https://github.com/wurenkai/HSH-UNet>, HEAD
  `341bbaa132177e3c1002e651050637d6a8f08130`.
- Se repitió la búsqueda de extensiones y enlaces. Como MHorUNet, el README pide
  proporcionar en `resume_model` un checkpoint generado localmente; no ofrece
  uno público.
- Decisión: **eliminado de la web**.

## SET (no implementado)

- Fuente: <https://github.com/Wzhjerry/SET>, HEAD
  `2f99bc1c2771bc634705e4ed78b7cab3e30f2147`.
- Se inspeccionaron el árbol, README, configuración y releases. Los checkpoints
  mencionados son salidas de entrenamiento bajo `save_name`. Además, el README
  conserva “Release pretrained weights of SET on ISIC2018” en la lista de
  tareas pendientes.
- Decisión: **eliminado de la web**.

## EM-Net (no implementado)

- Fuente: <https://github.com/Bean-Young/EM-Net>, HEAD
  `34b3cb4902c02d2a7f090ed45a4359c11ac94935`.
- La búsqueda encontró dos `.pth`, pero `find -printf '%s'` mostró que cada uno
  mide exactamente **1 byte**. El README aclara que son solo demostraciones y
  no deben usarse para pruebas; solicita pedir por correo los pesos de cada
  dataset.
- Un archivo privado entregado por correo no cumple reproducibilidad pública.
- Decisión: **eliminado de la web**.

## MobileSAM (no implementado en la comparación automática)

- Fuente: <https://github.com/ChaoningZhang/MobileSAM> y checkpoint local
  `models/mobile_sam/mobile_sam.pt` usado en el piloto previo.
- Se verificó que carga y produce tres candidatos por imagen con un punto
  positivo central y cuatro negativos en las esquinas. No produce una única
  máscara de lesión sin prompts ni una regla humana de selección.
- La revisión cualitativa documentó selección de fondo/piel, fallos en lesiones
  de bajo contraste, artefactos, lesiones múltiples y bordes irregulares. Elegir
  automáticamente el candidato de mayor score cambiaría el protocolo y no
  resolvería esos fallos.
- Decisión: **se conserva como piloto documentado, pero se elimina del selector
  de modelos supervisados sin prompts**. Véase
  `docs/MOBILE_SAM_PROMPT_PILOT_FINDINGS.md`.

## Reproducción de los modelos incluidos

Con `thesis-avit` creado, todos los instaladores/checkpoints se preparan así:

```bash
conda run --no-capture-output -n thesis-avit python scripts/setup_avit_model.py
conda run --no-capture-output -n thesis-avit python scripts/setup_ultralight_vm_unet.py
conda run --no-capture-output -n thesis-avit python scripts/setup_ba_transformer.py
conda run --no-capture-output -n thesis-avit \
  python -m pip install --no-deps segmentation-models-pytorch==0.5.0
conda run --no-capture-output -n thesis-avit python scripts/setup_unet_resnet34.py
conda run --no-capture-output -n thesis-avit python scripts/setup_skinmamba.py
mamba install -n thesis-avit -c conda-forge transformers
conda run --no-capture-output -n thesis-avit python scripts/setup_unixio_isic2018.py
conda run --no-capture-output -n thesis-avit python scripts/setup_theodore_isic2018.py
conda run --no-capture-output -n thesis-avit python scripts/setup_vmunet.py
conda run --no-capture-output -n thesis-avit python scripts/setup_delightsam.py
```

Los pesos y clones externos viven bajo `models/` y permanecen excluidos de Git;
los scripts, hashes, commits, adaptadores y este registro sí quedan versionados.
