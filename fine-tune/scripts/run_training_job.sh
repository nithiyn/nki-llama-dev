#!/usr/bin/env bash
set -e

cd ~/neuronx-distributed-training/examples
export CONF_FILE=hf_llama3_8B_SFT_lora_config
export COMPILE=0

bash train.sh
