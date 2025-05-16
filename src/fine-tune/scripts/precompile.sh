#!/usr/bin/env bash
set -e
echo "==== Starting NxDT model compilation script ===="

echo "==== Step 1: Navigating to NxDT examples directory ===="
# 1. Go to the NxDT repo you already cloned
cd ~/nki-llama/src/fine-tune/neuronx-distributed-training/examples
echo "Current directory: $(pwd)"

echo "==== Step 2: Copying configuration files ===="
# Copy your config files to the examples conf directory
cp -v ~/nki-llama/src/fine-tune/configs/YAML/*.yaml \
      ~/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/
echo "Configuration files copied successfully"

echo "==== Step 3: Setting configuration variables ===="
# Export config variables (order matters!)
export CONF_FILE=hf_llama3.1_8B_SFT_lora_config
export COMPILE=1      # tells train.sh to just pre‑compile graphs
echo "Configuration file set to: $CONF_FILE"
echo "COMPILE flag set to: $COMPILE (will perform AOT compilation only)"

echo "==== Step 4: Running AOT compilation ===="
# Run AOT compile
bash train.sh
echo "==== AOT compilation process complete! ===="