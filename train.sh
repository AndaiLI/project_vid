#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

dora run solver=VidMuse/finetune \
        continue_from=//pretrained/facebook/musicgen-small \
        model/lm/model_scale=small