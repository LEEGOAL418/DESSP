#!/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    mace_run_train \
    --config "./configs/MACE_ORB_Distilled.yaml" \
    --distributed
    
