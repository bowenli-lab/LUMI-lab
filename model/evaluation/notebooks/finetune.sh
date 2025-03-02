#!/bin/bash

cd $(dirname "$0") # dir of this script
conda activate unimol

exp_name="exp1102_export"

# 1. generate lmdb file
python finetune.py -f "${exp_name}.csv"

# 2. finetune models
export HF_DATASETS_CACHE="~/.cache/huggingface"
weight_name="continual-pretrain-a100_large_epoch-lr2e-5-cw10.0-lg0.5.new"

python unimol_huggingface_kfold.py \
    --weight-name ${weight_name} \
    --loss-type "weighted_mse" \
    --lmdb-data "${exp_name}.lmdb" \
    --save-model
