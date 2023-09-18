#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
DATA_PATH=/mnt/mlin/bloom/data/Dahoas/rm-static/data

LLAMA2_PATH=/mnt/mlin/llama-2
MODEL_PATH=$LLAMA2_PATH/tb-13b-base

CHECKPOINT_DIR=$LLAMA2_PATH/checkpoints/rm_tb-13b
OUTPUT=$LLAMA2_PATH/models/rm_tb-13b

ZERO_STAGE=2
LR=1e-5

deepspeed main.py \
   --data_path $DATA_PATH \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1000 \
   --learning_rate $LR \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --steps_per_print 1 \
   --checkpoint_steps 50 \
   --checkpoint_dir $CHECKPOINT_DIR \
   --output_dir $OUTPUT
