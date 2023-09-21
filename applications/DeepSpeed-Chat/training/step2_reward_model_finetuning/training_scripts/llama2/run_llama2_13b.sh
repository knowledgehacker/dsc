#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
DATA_PATH=/mnt/ai2/bloom/data/Dahoas/rm-static/data

MODEL_SIZE=13b

LLAMA2_PATH=/mnt/ai2/llama-2

MODEL_PATH=$LLAMA2_PATH/Llama-2-${MODEL_SIZE}-hf

CHECKPOINT_DIR=$LLAMA2_PATH/checkpoints/rm_Llama-2-${MODEL_SIZE}
OUTPUT=$LLAMA2_PATH/models/rm_raw-${MODEL_SIZE}

# TODO: we can compare the performance results of PPO by using reward model trained on continued pretrained model or SFT model
#MODEL_PATH=$LLAMA2_PATH/models/llama_pt

#CHECKPOINT_DIR=$LLAMA2_PATH/checkpoints/rm_tb-${MODEL_SIZE}
#OUTPUT=$LLAMA2_PATH/models/rm_tb-${MODEL_SIZE}

ZERO_STAGE=2
LR=1e-5

nohup sh -c "deepspeed --include localhost:0,1,2,3,4,5 --master_port 28900 main.py \
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
   --output_dir $OUTPUT > rm.log 2>&1" &

