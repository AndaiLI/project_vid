#!/bin/bash

# =================================================================
# FINAL, DEFINITIVE SMOKE TEST SCRIPT
# =================================================================
# This version removes the comments from within the multi-line command,
# which was the final, root cause of all our execution problems.

# 设置 Hugging Face 镜像站环境变量 (This is good practice)
export HF_ENDPOINT="https://hf-mirror.com"

echo "--- Starting Stylized VidMuse Fine-tuning Smoke Test (Final, Definitive) ---"

dora run solver=VidMuse/VidMuse_example \
    \
    conditioner=stylized_text2music \
    \
    '+continue_from_pretrained="/data/lw/musci_generation/VidMuse/VidMuse_v1/project_vid/model/state_dict.bin"' \
    \
    'optim.epochs=1' \
    'optim.updates_per_epoch=5' \
    'optim.lr=1e-5' \
    'optim.ema.use=false' \
    'schedule.lr_scheduler=null' \
    'evaluate.every=null' \
    'generate.every=null' \
    'checkpoint.save_every=null'

if [ $? -eq 0 ]; then
    echo "--- Dora run finished successfully. SMOKE TEST PASSED! ---"
else
    echo "--- Dora run failed. SMOKE TEST FAILED. ---"
fi