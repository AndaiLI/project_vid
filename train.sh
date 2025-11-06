#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

export CUDA_VISIBLE_DEVICES="5" 

dora run solver=VidMuse/finetune-v1 \
        continue_from=//pretrained/facebook/musicgen-small \
        model/lm/model_scale=small