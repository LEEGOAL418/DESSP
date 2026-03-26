#!/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ../mace/cli/run_train.py \
    --distributed \
    --config "./configs/MACE_VASPGO.yaml"
