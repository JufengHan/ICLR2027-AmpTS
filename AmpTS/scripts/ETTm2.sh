#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
  case ${pred_len} in
    96) restoration_steps=15 ;;
    192|336|720) restoration_steps=18 ;;
  esac

  python -u run.py \
    --is_training 1 \
    --data ETTm2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --pred_len ${pred_len} \
    --restoration_steps ${restoration_steps} \
    --history_mode conv_decomp \
    --conv_kernel 1 \
    --learning_rate 0.00005 \
    --train_epochs 10 \
    --patience 5 \
    --lradj type1 \
    --random_seed 2026
done