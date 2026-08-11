#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run.py train \
    --train-src=zh_en_data/train_1K.zh \
    --train-tgt=zh_en_data/train_1K.en \
    --dev-src=zh_en_data/dev_100.zh \
    --dev-tgt=zh_en_data/dev_100.en \
    --vocab=vocab.json \
    --cuda \
    --embed-size=128 \
    --hidden-size=128 \
    --batch-size=8 \
    --max-epoch=1 \
    --valid-niter=125 \
    --log-every=25 \
    --max-decoding-time-step=20 \
    --dropout=0.0 \
    --save-to=toy_model_colab.bin