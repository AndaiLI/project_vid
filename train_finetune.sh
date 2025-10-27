#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# 极简版VidMuse微调启动脚本

dora run solver=VidMuse/finetune \
        continue_from=/data/lw/musci_generation/VidMuse/VidMuse_v1/project_vid/model/state_dict.bin