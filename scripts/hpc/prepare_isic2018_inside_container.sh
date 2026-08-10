#!/usr/bin/env bash
# Download, import, decontaminate and verify official ISIC 2018 in one invocation.
set -Eeuo pipefail
DATA_ROOT=${1:?Pass the persistent ISIC data root}
MANIFEST_ROOT="$DATA_ROOT/manifests"
mkdir -p "$MANIFEST_ROOT" "$DATA_ROOT/metadata"

python scripts/benchmark/manage_datasets.py download-isic2018 --output "$DATA_ROOT" --confirm-download
python scripts/benchmark/manage_datasets.py extract-isic2018 --data-root "$DATA_ROOT"
for split in train validation test; do
    manifest="$MANIFEST_ROOT/isic2018_task1_${split}.json"
    python scripts/benchmark/manage_datasets.py import-isic2018 \
        --data-root "$DATA_ROOT" --split "$split" --output "$manifest"
    python scripts/benchmark/manage_datasets.py download-isic-metadata \
        --manifest "$manifest" --output "$DATA_ROOT/metadata" --confirm-download
    python scripts/benchmark/manage_datasets.py import-isic2018 \
        --data-root "$DATA_ROOT" --split "$split" --metadata-root "$DATA_ROOT/metadata" --output "$manifest"
    python scripts/benchmark/manage_datasets.py verify --manifest "$manifest" --data-root "$DATA_ROOT"
done

python scripts/benchmark/derive_isic2018_disjoint.py \
    --train "$MANIFEST_ROOT/isic2018_task1_train.json" \
    --validation "$MANIFEST_ROOT/isic2018_task1_validation.json" \
    --test "$MANIFEST_ROOT/isic2018_task1_test.json" \
    --output-root "$MANIFEST_ROOT" \
    --expected-train-count 2522 \
    --expected-validation-count 64 \
    --expected-test-count 1000

python scripts/benchmark/build_disjoint_folds.py \
    --manifest "$MANIFEST_ROOT/isic2018_task1_train_disjoint.json" \
    --output "$MANIFEST_ROOT/isic2018_task1_train_disjoint_folds_5.json" \
    --folds 5 --seed 20260806

python scripts/benchmark/manage_datasets.py verify \
    --manifest "$MANIFEST_ROOT/isic2018_task1_train_disjoint.json" --data-root "$DATA_ROOT"
python scripts/benchmark/manage_datasets.py verify \
    --manifest "$MANIFEST_ROOT/isic2018_task1_validation_disjoint.json" --data-root "$DATA_ROOT"

echo "ISIC 2018 disjoint preparation completed: train=2522 validation=64 test=1000; test masks remain sealed."
