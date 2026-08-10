#!/usr/bin/env bash
# Prepare external IMA++ v1.1 in one container invocation after license review.
set -Eeuo pipefail
IMAPP_ROOT=${1:?Pass the persistent IMA++ root}
mkdir -p "$IMAPP_ROOT/source" "$IMAPP_ROOT/manifests" "$IMAPP_ROOT/images" "$IMAPP_ROOT/image_metadata_api"
python scripts/benchmark/manage_datasets.py download-imapp-metadata \
    --output "$IMAPP_ROOT/source" --include-segmentations --confirm-download
python scripts/benchmark/manage_datasets.py import-imapp \
    --source-root "$IMAPP_ROOT/source" --data-root "$IMAPP_ROOT" --output "$IMAPP_ROOT/manifests"
for split in train validation test; do
    python scripts/benchmark/manage_datasets.py download-isic-images \
        --manifest "$IMAPP_ROOT/manifests/imapp_${split}.json" --output "$IMAPP_ROOT/images" \
        --metadata-output "$IMAPP_ROOT/image_metadata_api" --confirm-download
done
python scripts/benchmark/manage_datasets.py import-imapp \
    --source-root "$IMAPP_ROOT/source" --data-root "$IMAPP_ROOT" --output "$IMAPP_ROOT/manifests"
for split in train validation test; do
    python scripts/benchmark/manage_datasets.py verify \
        --manifest "$IMAPP_ROOT/manifests/imapp_${split}.json" --data-root "$IMAPP_ROOT"
done
echo "IMA++ v1.1 preparation completed for external evaluation only."
