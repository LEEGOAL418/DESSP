#!/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 mace_run_train \
    --config="./configs/MACE_DESSP.yaml" \
    --distributed
