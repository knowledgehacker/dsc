#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
DATA_PATH=/mnt/ai2/bloom/data/Dahoas/rm-static/data

MODEL_SIZE=13b

LLAMA2_PATH=/mnt/ai2/llama-2

ACTOR_MODEL_PATH=$LLAMA2_PATH/Llama-2-${MODEL_SIZE}-hf
CRITIC_MODEL_PATH=$LLAMA2_PATH/checkpoints/rm_raw-${MODEL_SIZE}

CHECKPOINT_DIR=$LLAMA2_PATH/checkpoints/rl_raw-${MODEL_SIZE}
OUTPUT=$LLAMA2_PATH/models/rl_raw-${MODEL_SIZE}

# actor model should use SFT model on continued pretrained model
# use meta Llama 2 13b chat for testing currently, since llama_mt can not be loaded using from_pretrained directly
#ACTOR_MODEL_PATH=$LLAMA2_PATH/models/llama_mt
#ACTOR_MODEL_PATH=$LLAMA2_PATH/Llama-2-13b-chat-hf
# critic model use reward model trained on continued pretrained model or SFT model
#CRITIC_MODEL_PATH=$LLAMA2_PATH/checkpoints/rm_tb-${MODEL_SIZE}

#CHECKPOINT_DIR=$LLAMA2_PATH/checkpoints/rl_tb-${MODEL_SIZE}
#OUTPUT=$LLAMA2_PATH/models/rl_tb-${MODEL_SIZE}

ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3

Actor_Lr=9.65e-6
Critic_Lr=5e-6


nohup sh -c "deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 28579 main.py \
   --data_path $DATA_PATH \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 4 \
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
   --gradient_accumulation_steps 4 \
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
   --output_dir $OUTPUT > rl.log 2>&1" &
