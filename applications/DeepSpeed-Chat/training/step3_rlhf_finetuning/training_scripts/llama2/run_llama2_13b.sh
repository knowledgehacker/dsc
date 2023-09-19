#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
DATA_PATH=/mnt/mlin/bloom/data/Dahoas/rm-static/data

ACTOR_MODEL_PATH=/mnt/mlin/llama-2/tb-13b-chat
CRITIC_MODEL_PATH=/mnt/mlin/llama-2/checkpoints/rm_tb-13b

ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3

Actor_Lr=9.65e-6
Critic_Lr=5e-6

OUTPUT=/mnt/mlin/llama-2/models/rl_tb-13b

deepspeed --include localhost:0,1 --master_port 28579 main.py \
   --data_path $DATA_PATH \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --seed 1234 \
   --deepspeed \
   --offload \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT
