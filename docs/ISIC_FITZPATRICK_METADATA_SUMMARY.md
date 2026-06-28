# ISIC-Fitzpatrick Metadata Summary

## Source

Metadata downloaded from ISIC Archive using `isic-cli` with this filter:

`fitzpatrick_skin_type:I OR fitzpatrick_skin_type:II OR fitzpatrick_skin_type:III OR fitzpatrick_skin_type:IV OR fitzpatrick_skin_type:V OR fitzpatrick_skin_type:VI`

Local raw file:

`data/raw/isic_fitzpatrick_metadata_full.csv`

The raw CSV is ignored by Git.

## Dataset size

- Rows: 12,335 images
- Columns: 33
- File size: 3.056 MB

## Fitzpatrick distribution

| Fitzpatrick type | Count |
|---|---:|
| I | 2,947 |
| II | 5,118 |
| III | 1,774 |
| IV | 922 |
| V | 823 |
| VI | 751 |

## Image type

| Image type | Count |
|---|---:|
| dermoscopic | 9,255 |
| clinical: close-up | 2,741 |
| clinical: overview | 339 |

## Diagnosis level 1

| Diagnosis | Count |
|---|---:|
| Benign | 9,859 |
| Malignant | 2,083 |
| Indeterminate | 393 |

## Copyright license

| License | Count |
|---|---:|
| CC-BY | 9,817 |
| CC-BY-NC | 1,915 |
| CC-0 | 603 |

## Main attributions

| Attribution | Count |
|---|---:|
| Memorial Sloan Kettering Cancer Center | 5,588 |
| Sydney Melanoma Diagnostic Center / Pascale Guitera | 1,915 |
| MEL-SELF Trial | 1,821 |
| Hospital Italiano de Buenos Aires | 1,517 |
| Federal University of Espirito Santo UFES | 1,494 |

## Patient and lesion identifiers

| Field | Non-null | Unique |
|---|---:|---:|
| isic_id | 12,335 | 12,335 |
| patient_id | 12,335 | 2,415 |
| lesion_id | 12,335 | 6,597 |

## Sex

| Sex | Count |
|---|---:|
| female | 6,598 |
| male | 5,705 |
| NaN | 32 |

## Anatomical site level 1

| Site | Count |
|---|---:|
| Trunk | 5,319 |
| Upper extremity | 2,682 |
| Lower extremity | 2,407 |
| Head and neck | 1,828 |
| NaN | 99 |

## Critical interpretation

This subset is usable for metadata auditing and controlled experiments, but it should not be treated as a single homogeneous dataset.

Important points:

1. The subset is strongly imbalanced toward Fitzpatrick I-II.
2. Fitzpatrick IV-VI are present in useful numbers but remain minority groups.
3. The subset mixes dermoscopic and clinical images, so modality must be controlled.
4. `patient_id` and `lesion_id` are available, so splits must avoid patient-level or lesion-level leakage.
5. The dataset supports binary benign/malignant/indeterminate analysis more directly than fine-grained multiclass classification.
6. Fine-grained labels such as `diagnosis_3` have substantial missingness and require filtering.
7. Segmentation masks generated with SAM, SAMed, MedSAM, or SAM-Med-style models should be validated before downstream classification.
