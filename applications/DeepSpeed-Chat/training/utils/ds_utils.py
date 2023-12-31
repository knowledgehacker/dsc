# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    # zero configuration
    if stage == 3:
        zero_opt_dict = {
            "stage": stage,
            "offload_optimizer": {
                "device": device
            },
            "offload_param": {
                "device": device
            },
            "overlap_comm": False,
            "contiguous_gradients": True,
            "sub_group_size": 5e8,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 5e8,
            "stage3_max_reuse_distance": 5e8,
            "stage3_gather_16bit_weights_on_model_save": True,

            "memory_efficient_linear": False
        }
    elif stage == 2:
        zero_opt_dict = {
            "stage": stage,
            "offload_optimizer": {
                "device": device,
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": False,
            "contiguous_gradients": True
        }
    else:
        print("Unsupported stage - %d" % stage)
        exit(-1)

    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != torch.cuda.device_count():
            zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count(
            )
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": device
        },

        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
