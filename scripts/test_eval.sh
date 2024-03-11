#!/usr/bin/env bash

print_usage() {
    echo "Usage: test_eval.sh [option1] [option2] [option3]"
    echo "   option1 - CKPT: path of model checkpoint."
    echo "   option2 - DATASET: 'VCDB' or 'VCSL'"
    echo "   option3 - FEAT: 'eff256d' or 'RTR'"
}

# help command
if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_usage
    exit 1
fi

CKPT=${1}
DATASET=${2:-"VCSL"}
FEAT=${3:-"eff256d"}
FEAT_DIR="features/${DATASET}/${FEAT}/"

echo "Set CKPT: ${CKPT}"
echo "Set DATASET: ${DATASET}"
echo "Set FEAT_DIR: ${FEAT_DIR}"

python run.py \
       --model-file ${CKPT} \
       --feat-dir ${FEAT_DIR} \
       --test-file data/${DATASET}/pair_file.csv \
       --save-file results/${DATASET}/result.json \
       --inference-batch 256

python evaluation.py \
       --anno-file data/${DATASET}/label_file.json \
       --test-file data/${DATASET}/pair_file.csv \
       --pred-file results/${DATASET}/result.json

echo "done!"
