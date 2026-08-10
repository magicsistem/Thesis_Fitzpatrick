#!/usr/bin/env bash
# Download, import, enrich and verify official ISIC 2018 in one container invocation.
set -Eeuo pipefail
DATA_ROOT=${1:?Pass the persistent ISIC data root}
mkdir -p "$DATA_ROOT/manifests" "$DATA_ROOT/metadata"

python scripts/benchmark/manage_datasets.py download-isic2018 --output "$DATA_ROOT" --confirm-download
python scripts/benchmark/manage_datasets.py extract-isic2018 --data-root "$DATA_ROOT"
for split in train validation test; do
    manifest="$DATA_ROOT/manifests/isic2018_task1_${split}.json"
    python scripts/benchmark/manage_datasets.py import-isic2018 \
        --data-root "$DATA_ROOT" --split "$split" --output "$manifest"
    python scripts/benchmark/manage_datasets.py download-isic-metadata \
        --manifest "$manifest" --output "$DATA_ROOT/metadata" --confirm-download
    python scripts/benchmark/manage_datasets.py import-isic2018 \
        --data-root "$DATA_ROOT" --split "$split" --metadata-root "$DATA_ROOT/metadata" --output "$manifest"
    python scripts/benchmark/manage_datasets.py verify --manifest "$manifest" --data-root "$DATA_ROOT"
done
python scripts/benchmark/build_folds.py --dataset isic2018_task1 --split train --folds 5 --seed 20260806 \
    --data-root "$DATA_ROOT" --manifest "$DATA_ROOT/manifests/isic2018_task1_train.json" \
    --output "$DATA_ROOT/manifests/isic2018_task1_train_folds_5.json"
python scripts/benchmark/manage_datasets.py audit-overlap \
    --manifest "$DATA_ROOT/manifests/isic2018_task1_train.json" \
    --manifest "$DATA_ROOT/manifests/isic2018_task1_validation.json" \
    --manifest "$DATA_ROOT/manifests/isic2018_task1_test.json" \
    --output "$DATA_ROOT/manifests/isic2018_task1_overlap_audit.json"
echo "ISIC 2018 preparation completed; sealed test masks remain unextracted."
