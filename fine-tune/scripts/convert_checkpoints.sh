#!/usr/bin/env bash
set -e
source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate

# Use NxDT’s checkpoint converter to produce sharded NxDT checkpoint
python -m neuronx_distributed_training.checkpoint_converter \
    --model_style hf \
    --hf_model_name ./llama3-8B_hf_weights \
    --output_dir ./pretrained_ckpt \
    --hw_backend trn1 \
    --config ./conf/8B_config_llama3/config.json
