#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --data ETTm1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --pred_len ${pred_len} \
    --restoration_steps 15 \
    --history_mode conv_decomp \
    --conv_kernel 1 \
    --learning_rate 0.00005 \
    --train_epochs 20 \
    --patience 5 \
    --lradj type2 \
    --random_seed 2026
done
