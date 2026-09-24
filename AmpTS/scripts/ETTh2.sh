#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
  case ${pred_len} in
    96)
      restoration_steps=5
      history_mode=conv_decomp
      conv_kernel=1
      moving_avg=17
      ;;
    192)
      restoration_steps=15
      history_mode=conv_decomp
      conv_kernel=1
      moving_avg=17
      ;;
    336)
      restoration_steps=4
      history_mode=decomp
      conv_kernel=3
      moving_avg=17
      ;;
    720)
      restoration_steps=6
      history_mode=decomp
      conv_kernel=1
      moving_avg=7
      ;;
  esac

  python -u run.py \
    --is_training 1 \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --pred_len ${pred_len} \
    --restoration_steps ${restoration_steps} \
    --history_mode ${history_mode} \
    --conv_kernel ${conv_kernel} \
    --moving_avg ${moving_avg} \
    --learning_rate 0.0001 \
    --train_epochs 20 \
    --patience 5 \
    --lradj type1 \
    --random_seed 2025
done
