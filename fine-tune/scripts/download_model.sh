#!/usr/bin/env bash
set -e

# Paths
TOKENIZER_DIR=./llama3_tokenizer
MODEL_DIR=./llama3-8B_hf_weights

# 1. Download from Hugging Face via Python (so we can save in HF style)
python - <<'PYCODE'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "meta-llama/Meta-Llama-3-8B"
AutoTokenizer.from_pretrained(model_id).save_pretrained("llama3_tokenizer")
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained("llama3-8B_hf_weights")
PYCODE

# 2. Convert any .safetensors shards to classic .bin
python scripts/convert_safetensors.py --input_dir ${MODEL_DIR} --output_dir ${MODEL_DIR}
