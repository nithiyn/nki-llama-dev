#!/usr/bin/env bash
set -e

echo "==== Starting Llama model download and conversion script ===="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set!"
    echo "Please make sure HF_TOKEN is defined in your .env file"
    exit 1
fi

echo "==== Changing to fine-tune workspace ===="
# Go to your fine-tune workspace
cd ~/nki-llama/src/fine-tune
echo "Current directory: $(pwd)"

echo "==== Setting path variables ===="
# Paths
export TOKENIZER_DIR=~/nki-llama/src/fine-tune/model_assets/llama_tokenizer
export MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/llama_3-1_8b
export BIN_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/llama3-8B_hf_weights_bin
export CONSOLIDATED_BIN_MODEL_DIR=~/nki-llama/src/fine-tune/model_assets/pckpt/

echo "Tokenizer directory: $TOKENIZER_DIR"
echo "Model directory: $MODEL_DIR"
echo "Binary model directory: $BIN_MODEL_DIR"
echo "Consolidated binary model directory: $CONSOLIDATED_BIN_MODEL_DIR"

echo "==== Creating output directories ===="
# Create the output directories if needed
mkdir -p "${TOKENIZER_DIR}" "${MODEL_DIR}" "${BIN_MODEL_DIR}" "${CONSOLIDATED_BIN_MODEL_DIR}"
echo "Output directories created"

echo "==== Step 1: Downloading model and tokenizer from Hugging Face ===="
# 1. Download from Hugging Face via Python (so we can save in HF style)
python - <<'PYCODE'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer_dir = os.path.expanduser(os.environ["TOKENIZER_DIR"])
model_dir     = os.path.expanduser(os.environ["MODEL_DIR"])
hf_token      = os.environ.get("HF_TOKEN")

print(f"Downloading tokenizer from {model_id}...")
AutoTokenizer.from_pretrained(model_id, token=hf_token).save_pretrained(tokenizer_dir)
print(f"Tokenizer saved to {tokenizer_dir}")

print(f"Downloading model from {model_id}...")
AutoModelForCausalLM.from_pretrained(model_id, token=hf_token).save_pretrained(model_dir)
print(f"Model saved to {model_dir}")
PYCODE

echo "==== Step 2: Converting .safetensors to .bin format ===="
# 2. Convert any .safetensors shards to classic .bin
python ~/nki-llama/src/fine-tune/scripts/convert_safetensors.py \
    --input_dir  "${MODEL_DIR}" \
    --output_dir "${BIN_MODEL_DIR}"
echo "Conversion to .bin format complete"

echo "==== Step 3: Merging checkpoints into a single .bin file ===="
# 3. needs to be one .bin file
python ~/nki-llama/src/fine-tune/scripts/merge_checkpoints.py
echo "Checkpoint merging complete"

echo "==== Model download and conversion process complete! ===="