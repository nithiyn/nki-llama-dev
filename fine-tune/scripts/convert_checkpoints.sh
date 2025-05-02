#!/usr/bin/env bash
set -e

cd ~/nki-llama/fine-tune/

git clone https://github.com/aws-neuron/neuronx-distributed-training.git
cd ~/nki-llama/fine-tune/neuronx-distributed-training/examples/checkpoint_converter_scripts
# Use NxDT’s checkpoint converter to produce sharded NxDT checkpoint
export PCKPT_MODEL_DIR=~/nki-llama/fine-tune/model_assets/converted_hf_style_hf_to_nxdt_tp8pp4/
export MODEL_CONFIG=~/nki-llama/fine-tune/configs/model-config/config.json
export BIN_MODEL_DIR=~/nki-llama/fine-tune/model_assets/pckpt/pytorch_model.bin"
python3 checkpoint_converter.py \
    --model_style             hf \
    --hw_backend              trn1 \
    --input_dir               $BIN_MODEL_DIR \
    --output_dir              $PCKPT_MODEL_DIR \
    --save_xser               True \
    --config                  $MODEL_CONFIG \
    --tp_size                 32 \
    --pp_size                 1 \
    --n_layers                32 \
    --kv_size_multiplier      4 \
    --qkv_linear              True \
    --convert_from_full_state
