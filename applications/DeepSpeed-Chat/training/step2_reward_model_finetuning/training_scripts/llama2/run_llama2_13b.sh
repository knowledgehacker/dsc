#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
DATA_PATH=/mnt/mlin/bloom/data/Dahoas/rm-static/data

LLAMA2_PATH=/mnt/mlin/llama-2
MODEL_PATH=$LLAMA2_PATH/tb-13b-base

OUTPUT=$LLAMA2_PATH/rm_tb-13b

ZERO_STAGE=3
LR=1e-5

deepspeed main.py \
   --data_path $DATA_PATH \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate $LR \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --output_dir $OUTPUT
